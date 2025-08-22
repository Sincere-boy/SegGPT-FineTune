import os
import json
import time
import numpy as np
import copy
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
from transformers import SegGptImageProcessor, SegGptForImageSegmentation

# ======================
# 可配参数
# ======================
INPUT_DIR   = "images/test_dir"  # 用于统计的图片目录
OUTPUT_DIR  = "result/test_dir"  # 统计结果保存目录
PROMPT_DIR  = "prompt_images/ball"  # 提示图与掩码所在目录
CHECKPOINT  = "checkpoint/seggpt-vit-large"

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_ENSEMBLE = True                   # 与线上一致时统计更可信
MAX_CALIB   = 50                          # 最多统计多少张图片（可调，大一点更稳）

# ======================
# 通用工具
# ======================
def log(msg: str):
    print(f"[SegGPT-Stats] {msg}")

def list_images(folder: str) -> List[str]:
    exts = (".jpeg", ".jpg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]

def to_bs_c(x: torch.Tensor) -> torch.Tensor:
    """
    把隐藏态统一成 (B*S, C) 以便计算余弦相似度。
    支持 (B, Hp, Wp, C) 或 (B, S, C) 两种。
    """
    if x.dim() == 4:  # (B, Hp, Wp, C)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
    elif x.dim() == 3:  # (B, S, C)
        pass
    else:
        raise ValueError(f"Unexpected hidden state shape: {x.shape}")
    B, S, C = x.shape
    return x.reshape(B * S, C)

def nhwc_adapt_to(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    把 NHWC 的张量，通过自适应池化调整到目标 H×W（通道与批大小不变）。
    """
    assert x.dim() == 4
    x = x.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
    x = F.adaptive_avg_pool2d(x, output_size=(H, W))
    x = x.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC
    return x

def align_for_similarity(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对齐同一 block 的输入 a、输出 b 的形状。
    正常情况下两者形状一致；若不一致，做稳健对齐以避免报错：
      - 4D NHWC：对 b 做自适应池化到 a 的 (H, W)；
      - 3D (B, S, C)：截断到相同的最小 S。
    """
    if a.dim() == 4 and b.dim() == 4:
        Ba, Ha, Wa, Ca = a.shape
        Bb, Hb, Wb, Cb = b.shape
        assert Ba == Bb and Ca == Cb, f"Batch/Channel mismatch: {a.shape} vs {b.shape}"
        if (Ha, Wa) != (Hb, Wb):
            b = nhwc_adapt_to(b, Ha, Wa)
        return a, b

    if a.dim() == 3 and b.dim() == 3:
        Ba, Sa, Ca = a.shape
        Bb, Sb, Cb = b.shape
        assert Ba == Bb and Ca == Cb, f"Batch/Channel mismatch: {a.shape} vs {b.shape}"
        if Sa != Sb:
            S = min(Sa, Sb)
            a = a[:, :S, :]
            b = b[:, :S, :]
        return a, b

    # 兜底：不同维度时，全部展平成 (B*S, C) 后再对齐到最小长度
    A = to_bs_c(a)
    B = to_bs_c(b)
    S = min(A.size(0), B.size(0))
    return A[:S], B[:S]

def cosine_mean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    计算输入/输出的平均余弦相似度（逐 token 后求均值）。
    内部做归一化，数值稳健。
    """
    a, b = align_for_similarity(a, b)
    A = to_bs_c(a)
    B = to_bs_c(b)
    A = F.normalize(A, dim=-1, eps=1e-6)
    B = F.normalize(B, dim=-1, eps=1e-6)
    return (A * B).sum(dim=-1).mean()

# ======================
# Hook 采集器（修复：稳健提取 Tensor）
# ======================
def extract_tensor(obj: Any) -> Optional[torch.Tensor]:
    """
    从常见的输出结构中稳健地提取代表 hidden state 的 Tensor：
    - Tensor: 直接返回
    - tuple/list: 返回第一个能递归取到 Tensor 的元素
    - 有 'last_hidden_state' 属性: 取之
    - dict: 优先尝试这些键: 'last_hidden_state', 'hidden_states', 'out', 'output', 0
    取不到就返回 None。
    """
    if isinstance(obj, torch.Tensor):
        return obj
    # ModelOutput (transformers) 通常可当 dict 处理
    if isinstance(obj, (list, tuple)):
        for it in obj:
            t = extract_tensor(it)
            if t is not None:
                return t
        return None
    if isinstance(obj, dict):
        for k in ("last_hidden_state", "hidden_states", "out", "output", 0, "x"):
            if k in obj:
                t = extract_tensor(obj[k])
                if t is not None:
                    return t
        # fallback: 任意值里找一个 tensor
        for v in obj.values():
            t = extract_tensor(v)
            if t is not None:
                return t
        return None
    # 有属性的对象（如 ModelOutput）
    for attr in ("last_hidden_state", "hidden_states", "out", "output"):
        if hasattr(obj, attr):
            t = extract_tensor(getattr(obj, attr))
            if t is not None:
                return t
    return None

class BlockIOCollector:
    """
    给每个 block 同时挂 pre-hook / forward-hook，
    收集每次 forward 的 (x_in, x_out) 对。
    修复点：对 tuple/ModelOutput 进行稳健的张量提取。
    """
    def __init__(self, blocks: nn.ModuleList):
        self.blocks = blocks
        self.io_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._handles = []
        self._caches: List[Dict[str, Optional[torch.Tensor]]] = []

    def _register(self):
        self._handles.clear()
        self._caches = [{"in": None, "out": None} for _ in self.blocks]

        for i, blk in enumerate(self.blocks):
            def pre_hook(module, inputs, idx=i):
                # 对 inputs 进行稳健提取；通常 inputs[0] 就是 hidden_states
                x_in = None
                if isinstance(inputs, tuple) and len(inputs) > 0:
                    x_in = extract_tensor(inputs[0])
                if x_in is None:
                    x_in = extract_tensor(inputs)
                if x_in is not None:
                    self._caches[idx]["in"] = x_in.detach()

            def fwd_hook(module, inputs, output, idx=i):
                # output 可能是 tuple/ModelOutput，需要提取第一主输出
                x_out = extract_tensor(output)
                if x_out is not None:
                    self._caches[idx]["out"] = x_out.detach()

            self._handles.append(blk.register_forward_pre_hook(pre_hook))
            self._handles.append(blk.register_forward_hook(fwd_hook))

    def __enter__(self):
        self._register()
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def collect_once(self):
        """
        在一次 forward 结束后，把缓存中的 (in, out) 依次取出，形成列表。
        然后清空缓存，准备下一次 forward。
        """
        self.io_pairs = []
        for cache in self._caches:
            x_in, x_out = cache["in"], cache["out"]
            self.io_pairs.append((x_in, x_out))
            cache["in"] = None
            cache["out"] = None
        return self.io_pairs

# ======================
# 寻找 encoder blocks
# ======================
def find_encoder_blocks(model: nn.Module) -> Tuple[nn.Module, str, nn.ModuleList]:
    """
    自动定位 SegGPT 的编码器 blocks（ModuleList）。
    常见路径：model.seggpt.encoder.blocks / model.model.encoder.blocks / ... / .layers / .layer
    """
    candidates = [getattr(model, "seggpt", None), getattr(model, "model", None), model]
    for root in candidates:
        if root is None:
            continue
        enc = getattr(root, "encoder", root)
        for name in ["blocks", "layers", "layer"]:
            if hasattr(enc, name):
                blocks = getattr(enc, name)
                if isinstance(blocks, nn.ModuleList) and len(blocks) > 0:
                    return enc, name, blocks
    raise RuntimeError("Cannot locate encoder blocks ModuleList. Use print(model) to inspect structure.")
# ======================
# 裁剪函数
# ======================
def prune_model(model, blocks, sims_mean, prune_ratio=0.3, keep_first=2, keep_last=2):
    num_blocks = len(blocks)
    k_prune = int(num_blocks * prune_ratio)
    indices = list(range(num_blocks))
    # 相似度高 -> 更冗余
    sorted_by_sim = sorted(indices, key=lambda i: (sims_mean[i] if sims_mean[i] is not None else -1), reverse=True)
    protected = set(list(range(keep_first)) + list(range(num_blocks - keep_last, num_blocks)))
    prune_candidates = [i for i in sorted_by_sim if i not in protected]
    prune_indices = set(prune_candidates[:k_prune])
    keep_indices  = [i for i in indices if i not in prune_indices]
    pruned_model = copy.deepcopy(model)
    # 重建 ModuleList
    encoder = getattr(pruned_model, "seggpt", None) or getattr(pruned_model, "model", None) or pruned_model
    encoder = getattr(encoder, "encoder", encoder)
    blocks_attr = None
    for name in ["layers", "blocks", "layer"]:
        if hasattr(encoder, name):
            blocks_attr = name
            old_blocks = getattr(encoder, name)
            break
    assert blocks_attr is not None and isinstance(old_blocks, nn.ModuleList)
    kept_blocks = [old_blocks[i] for i in keep_indices]
    setattr(encoder, blocks_attr, nn.ModuleList(kept_blocks))
    # 更新层数
    if hasattr(pruned_model.config, "num_hidden_layers"):
        pruned_model.config.num_hidden_layers = len(keep_indices)
    # 🔴 关键：重映射中间层索引，保持个数不变
    if hasattr(pruned_model.config, "intermediate_hidden_state_indices") and pruned_model.config.intermediate_hidden_state_indices:
        orig_inter = list(pruned_model.config.intermediate_hidden_state_indices)
        new_inter  = remap_intermediate_indices_after_prune(keep_indices, orig_inter)
        pruned_model.config.intermediate_hidden_state_indices = new_inter
        print("[Prune] remapped intermediate indices:", orig_inter, "->", new_inter)
    pruned_model.eval()
    return pruned_model, keep_indices, prune_indices

def remap_intermediate_indices_after_prune(
        keep_indices,                   # List[int]  剪后保留的“原始层号”，升序
        orig_inter_indices,             # List[int]  剪前 config.intermediate_hidden_state_indices
) -> list:
    """
    把“原始层号的中间层索引”映射到“剪后模型中的层号”(即保留下来层在新ModuleList中的位置)。
    若某个原索引被删除，则选择距离最近的保留层替代。
    返回：剪后模型的 indices（仍保持与原来相同的个数）。
    """
    # 原始层号 -> 新位置（0..len(keep)-1）
    pos_in_keep = {orig_idx: new_i for new_i, orig_idx in enumerate(keep_indices)}
    def nearest_kept(orig_idx):
        # 找与 orig_idx 距离最近的一个 keep_indices 元素
        return min(keep_indices, key=lambda k: abs(k - orig_idx))
    new_indices = []
    for oi in orig_inter_indices:
        if oi in pos_in_keep:
            new_indices.append(pos_in_keep[oi])
        else:
            nearest_orig = nearest_kept(oi)
            new_indices.append(pos_in_keep[nearest_orig])
    # 可能会出现重复（多个原索引映到同一个新层）；decoder 一般只要求个数和维度一致，
    # 允许重复并不影响尺寸。如果你想尽量去重，可在此做轻微微调，但务必保持个数不变。
    return new_indices

# ======================
# 主流程
# ======================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log(f"Using device: {DEVICE}")

    # 载入处理器与模型
    image_processor = SegGptImageProcessor.from_pretrained(CHECKPOINT, local_files_only=True)
    model = SegGptForImageSegmentation.from_pretrained(CHECKPOINT, local_files_only=True)
    model.to(DEVICE).eval()

    # 准备提示图与掩码
    prompt_image_path = os.path.join(PROMPT_DIR, "save4_2.jpeg")
    prompt_mask_path  = os.path.join(PROMPT_DIR, "save4_2_mask.jpeg")
    if not (os.path.exists(prompt_image_path) and os.path.exists(prompt_mask_path)):
        raise FileNotFoundError("Prompt image/mask not found in PROMPT_DIR.")

    prompt_images = [Image.open(prompt_image_path)]
    prompt_masks  = [Image.open(prompt_mask_path).convert("L")]

    # 找到 blocks
    encoder, blocks_name, blocks = find_encoder_blocks(model)
    num_blocks = len(blocks)
    log(f"Found encoder ModuleList '{blocks_name}' with {num_blocks} blocks.")

    # 统计容器
    layer_sims_sum = torch.zeros(num_blocks, dtype=torch.float64, device=DEVICE)
    layer_counts   = torch.zeros(num_blocks, dtype=torch.int32, device=DEVICE)

    # 待统计图片
    all_imgs = list_images(INPUT_DIR)
    if len(all_imgs) == 0:
        raise FileNotFoundError(f"No images in {INPUT_DIR}")
    if MAX_CALIB > 0:
        all_imgs = all_imgs[:MAX_CALIB]

    log(f"Calibrating with {len(all_imgs)} images ...")
    t0 = time.time()

    with torch.no_grad():
        with BlockIOCollector(blocks) as collector:
            for idx, img_path in enumerate(all_imgs, 1):
                img = Image.open(img_path)
                inputs = image_processor(
                    images=img,
                    prompt_images=prompt_images,
                    prompt_masks=prompt_masks,
                    return_tensors="pt",
                    feature_ensemble=FEATURE_ENSEMBLE
                )
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

                # 前向：只为触发 hooks，不需要取 outputs
                _ = model(**inputs)

                io_pairs = collector.collect_once()
                if len(io_pairs) != num_blocks:
                    log(f"Warning: collected {len(io_pairs)} IO pairs, expect {num_blocks}")

                per_image_sims = []
                for l, (x_in, x_out) in enumerate(io_pairs):
                    if (x_in is None) or (x_out is None):
                        per_image_sims.append(None)
                        continue
                    try:
                        sim = cosine_mean(x_in, x_out)
                        layer_sims_sum[l] += sim.to(layer_sims_sum.dtype)
                        layer_counts[l]   += 1
                        per_image_sims.append(float(sim.detach().cpu()))
                    except Exception as e:
                        log(f"[img#{idx}] Skip layer {l} due to: {repr(e)}")
                        per_image_sims.append(None)

                log(f"[{idx:>3}/{len(all_imgs)}] {os.path.basename(img_path)} "
                    f"per-block sims (sample): {per_image_sims[:min(6, num_blocks)]}")

    # 汇总
    sims_mean = []
    for l in range(num_blocks):
        if layer_counts[l].item() > 0:
            sims_mean.append(float((layer_sims_sum[l] / layer_counts[l]).detach().cpu()))
        else:
            sims_mean.append(None)

    # 输出与保存
    log("=== Per-block mean cosine similarity ===")
    for i, v in enumerate(sims_mean):
        log(f"Block {i:02d}: {v if v is not None else 'N/A'}")

    stats = {
        "checkpoint": CHECKPOINT,
        "num_blocks": num_blocks,
        "feature_ensemble": FEATURE_ENSEMBLE,
        "images_used": len(all_imgs),
        "cosine_mean_per_block": sims_mean,
    }
    out_json = os.path.join(OUTPUT_DIR, "block_cosine_stats.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    t1 = time.time()
    log(f"Saved stats to {out_json}")
    log(f"Total time: {t1 - t0:.2f}s")

    # 裁剪 30% 的 block
    pruned_model, keep_indices, prune_indices = prune_model(
        model, blocks, sims_mean, prune_ratio=0.4, keep_first=2, keep_last=2
    )

    pruned_dir = "result/test_dir/pruned_seggpt_40"
    os.makedirs(pruned_dir, exist_ok=True)
    pruned_model.save_pretrained(pruned_dir)
    image_processor.save_pretrained(pruned_dir)  # 方便离线一并加载
    with open(os.path.join(pruned_dir,"keep_indices.json"),"w") as f:
        json.dump({"keep_indices": keep_indices}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()