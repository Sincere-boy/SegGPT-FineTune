# infer_with_pruned_cfg.py  (pruned-dir only; no --ckpt needed)

import sys
sys.path.append('SegGPT/SegGPT_inference')

import os, json, argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from functools import partial

from SegGPT.SegGPT_inference.models_seggpt import SegGPT
from hf_keymap import map_hf_key_to_custom, looks_like_hf  # ⬅️ 用于 HF→自定义键名映射

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BIN_COLOR_MAP = np.array([[0, 0, 0], [0, 255, 255]], dtype=np.uint8)
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# ---------- utils ----------
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def to_tensor_nchw(np_img):
    x = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0).contiguous()
    return x

def norm_imagenet(np_img01):
    return (np_img01 - IMAGENET_MEAN) / IMAGENET_STD

def make_bool_mask_half(num_patches):
    m = torch.zeros(num_patches, dtype=torch.bool)
    m[num_patches // 2:] = True
    return m.unsqueeze(0)

def grayscale_mask_to_rgb01(mask_img):
    if mask_img.mode != "L":
        mask_img = mask_img.convert("L")
    m = np.array(mask_img, dtype=np.uint8)
    m_bin = (m > 127).astype(np.float32)
    m3 = np.stack([m_bin, m_bin, m_bin], axis=-1)
    return m3

def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in SUPPORTED_EXTS

# ---------- indices resolving ----------
def resolve_intermediate_indices(inter_cfg, keep_indices, num_layers_pruned):
    inter = [int(i) for i in inter_cfg]
    keep_set = set(keep_indices)
    if all(0 <= i < num_layers_pruned for i in inter) and not set(inter).issubset(keep_set):
        return inter, "pruned"
    pos_in_keep = {orig_idx: new_i for new_i, orig_idx in enumerate(keep_indices)}
    def nearest_kept(orig_idx):
        return min(keep_indices, key=lambda k: abs(k - orig_idx))
    result, used = [], set()
    for oi in inter:
        cand = pos_in_keep[oi] if oi in pos_in_keep else pos_in_keep[nearest_kept(oi)]
        if cand in used:
            unused = [k for k in keep_indices if pos_in_keep[k] not in used]
            if unused:
                cand = pos_in_keep[min(unused, key=lambda k: abs(k - oi))]
        used.add(cand)
        result.append(cand)
    return result, "orig"

# ---------- build model from pruned cfg ----------
def build_model_from_pruned_cfg(pruned_dir, device):
    cfg = load_json(os.path.join(pruned_dir, "config.json"))
    keep = load_json(os.path.join(pruned_dir, "keep_indices.json"))["keep_indices"]

    num_layers = int(cfg.get("num_hidden_layers", len(keep)))
    embed_dim   = int(cfg.get("hidden_size", 1024))
    num_heads   = int(cfg.get("num_attention_heads", 16))
    mlp_ratio   = float(cfg.get("mlp_ratio", 4.0))
    drop_path   = float(cfg.get("drop_path_rate", 0.0))
    use_rel     = bool(cfg.get("use_relative_position_embeddings", True))
    img_size    = tuple(cfg.get("image_size", [896, 448]))
    patch_size  = int(cfg.get("patch_size", 16))
    qkv_bias    = bool(cfg.get("qkv_bias", True))
    pre_imsize  = int(cfg.get("pretrain_image_size", 224))
    decoder_dim = int(cfg.get("decoder_hidden_size", 64))
    inter_cfg   = cfg.get("intermediate_hidden_state_indices", [5, 11, 17, 23])

    inter_after, mode = resolve_intermediate_indices(inter_cfg, keep, num_layers)
    print(f"[pruned-cfg] keep_indices(original-space) = {keep}")
    print(f"[pruned-cfg] intermediate indices mode={mode}; {inter_cfg} -> {inter_after}")

    model = SegGPT(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=cfg.get("num_channels", 3),
        embed_dim=embed_dim,
        depth=num_layers,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop_path_rate=drop_path,
        norm_layer=partial(torch.nn.LayerNorm, eps=cfg.get("layer_norm_eps", 1e-6)),
        act_layer=torch.nn.GELU,
        use_abs_pos=True,
        use_rel_pos=use_rel,
        rel_pos_zero_init=True,
        window_size=14,
        window_block_indexes=(),
        residual_block_indexes=[],
        use_act_checkpoint=False,
        pretrain_img_size=pre_imsize,
        pretrain_use_cls_token=True,
        out_feature="last_feat",
        decoder_embed_dim=decoder_dim,
        loss_func="smoothl1",
    ).to(device).eval()

    setattr(model, "intermediate_hidden_state_indices", tuple(inter_after))

    # decoder 输入维 sanity check
    expect_in = embed_dim * len(inter_after)
    if hasattr(model, "decoder_embed") and isinstance(model.decoder_embed, torch.nn.Linear):
        if model.decoder_embed.in_features != expect_in:
            raise RuntimeError(
                f"decoder_embed.in_features={model.decoder_embed.in_features} "
                f"but expect {expect_in} (=embed_dim*#intermediate_layers)."
            )
    return model, inter_after

# ---------- load HF weights from pruned_dir ----------
def load_weights_from_pruned_dir(model, pruned_dir):
    safep = os.path.join(pruned_dir, "model.safetensors")
    binp  = os.path.join(pruned_dir, "pytorch_model.bin")
    if os.path.isfile(safep):
        from safetensors.torch import load_file as load_safetensors
        sd = load_safetensors(safep, device="cpu")
    elif os.path.isfile(binp):
        sd = torch.load(binp, map_location="cpu")
    else:
        raise FileNotFoundError(f"Neither model.safetensors nor pytorch_model.bin found in {pruned_dir}")

    # HF -> 自定义键名
    if looks_like_hf(sd.keys()):
        mapped = {}
        for k, v in sd.items():
            nk = map_hf_key_to_custom(k)
            if nk is not None:
                mapped[nk] = v
        sd = mapped

    # pos_embed 兜底：CLS 长度差 1
    msd = model.state_dict()
    if "pos_embed" in sd and "pos_embed" in msd:
        pe = sd["pos_embed"]
        need = msd["pos_embed"].shape[1]
        have = pe.shape[1]
        if have != need and abs(have - need) == 1:
            if have > need:
                sd["pos_embed"] = pe[:, 1:, :]
            else:
                pad = torch.zeros(pe.shape[0], 1, pe.shape[2], dtype=pe.dtype)
                sd["pos_embed"] = torch.cat([pad, pe], dim=1)

    # 只加载匹配形状
    filtered = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
    missing = [k for k in msd.keys() if k not in filtered]
    unexpected = [k for k in sd.keys() if k not in msd]

    model.load_state_dict(filtered, strict=False)

    print(f"[hf-load] loaded={len(filtered)} / model_params={len(msd)}")
    if missing:
        print(f"[hf-load] missing={len(missing)}  e.g. {missing[:10]}")
    if unexpected:
        print(f"[hf-load] unexpected={len(unexpected)}  e.g. {unexpected[:10]}")
    return model

# ---------- single-prompt inference ----------
@torch.no_grad()
def run_one_image_single_prompt(model, inter_after, device, image_path, prompt_image_path, prompt_mask_path, out_path):
    res, hres = 448, 448
    im  = Image.open(image_path).convert("RGB").resize((res, hres))
    pim = Image.open(prompt_image_path).convert("RGB").resize((res, hres))
    pm  = Image.open(prompt_mask_path).convert("L").resize((res, hres), Image.NEAREST)

    img_np01 = np.asarray(im, dtype=np.float32) / 255.0
    p_np01   = np.asarray(pim, dtype=np.float32) / 255.0
    pm_rgb01 = grayscale_mask_to_rgb01(pm).astype(np.float32)

    tgt2 = pm_rgb01
    img2 = p_np01
    tgt = np.concatenate([tgt2, tgt2], axis=0)       # (2H, W, 3)
    img = np.concatenate([img2, img_np01], axis=0)   # (2H, W, 3)

    img = norm_imagenet(img)
    tgt = norm_imagenet(tgt)

    x = to_tensor_nchw(img).to(device).float()
    y = to_tensor_nchw(tgt).to(device).float()

    H, W = 2 * hres, res
    num_patches = (H // 16) * (W // 16)
    bool_masked_pos = make_bool_mask_half(num_patches).to(device)
    valid = torch.ones_like(y)
    seg_type = torch.zeros([1, 1]).to(device)  # semantic=0
    feat_ensemble = -1

    loss, pred_patchified, _ = model(x, y, bool_masked_pos, valid.float(), seg_type, feat_ensemble)
    pred = model.unpatchify(pred_patchified).permute(0, 2, 3, 1).detach().cpu()  # (1,2H,W,3)

    pred_bottom = pred[:, pred.shape[1] // 2 :, :, :]
    pred_bottom = torch.clip((pred_bottom * IMAGENET_STD + IMAGENET_MEAN) * 255.0, 0, 255)

    gray = pred_bottom[0].mean(dim=-1).numpy()
    bin_mask = (gray >= 128).astype(np.uint8)
    color = BIN_COLOR_MAP[bin_mask]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(color).save(out_path)
    print(f"[inference] saved to: {out_path}")

# ---------- CLI / batch ----------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pruned-dir", type=str, required=True, help="包含 config.json / keep_indices.json / model.safetensors 的目录")
    p.add_argument("--image", type=str, required=True, help="待分割图（文件或目录）")
    p.add_argument("--prompt-image", type=str, required=True, help="单张 prompt 图")
    p.add_argument("--prompt-mask", type=str, required=True, help="prompt 的灰度掩码(0/255)")
    p.add_argument("--out", type=str, default="", help="单图输出路径，如 outputs/pred.png；若 --image 是目录则忽略")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()

def process_path(model, inter_after, device, image_path: str, prompt_image_path: str, prompt_mask_path: str, out_path: str, pruned_dir: str):
    p = Path(image_path)
    if p.is_file():
        if not out_path:
            raise ValueError("单图推理需提供 --out 输出路径")
        run_one_image_single_prompt(model, inter_after, device, str(p), prompt_image_path, prompt_mask_path, out_path)
        return

    if p.is_dir():
        out_dir = Path("outputs") / Path(pruned_dir).name
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted([q for q in p.iterdir() if is_image_file(q)])
        if not files:
            raise FileNotFoundError(f"目录 {p} 中未找到图片：支持扩展名 {sorted(SUPPORTED_EXTS)}")
        for img_p in files:
            out_name = img_p.stem + "_pred.png"
            out_file = out_dir / out_name
            run_one_image_single_prompt(model, inter_after, device, str(img_p), prompt_image_path, prompt_mask_path, str(out_file))
        print(f"[batch] done. results saved under: {out_dir}")
        return

    raise FileNotFoundError(f"未找到路径：{image_path}")

def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1) 基于 pruned 配置构建模型
    model, inter_after = build_model_from_pruned_cfg(args.pruned_dir, device)
    # 2) 从 pruned_dir 加载 HF 权重
    model = load_weights_from_pruned_dir(model, args.pruned_dir).to(device).eval()
    # 3) 处理单/批
    with torch.no_grad():
        process_path(model, inter_after, device, args.image, args.prompt_image, args.prompt_mask, args.out, args.pruned_dir)

if __name__ == "__main__":
    main()

'''
python infer_with_hf.py \
  --pruned-dir pruned/seggpt-vit-large \
  --image data/0820/shui1_1.png \
  --prompt-image data/train/images/shui1_0.png \
  --prompt-mask  data/train/labels/shui1_0.png \
  --out outputs/pred_30_2.png \
  --device cuda
'''