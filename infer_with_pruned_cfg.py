# infer_with_pruned_cfg.py
import sys
sys.path.append('SegGPT/SegGPT_inference')

import os, json, argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from SegGPT.SegGPT_inference.models_seggpt import seggpt_vit_large_patch16_input896x448


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 简单二类配色：背景(0)=黑，前景(1)=青
BIN_COLOR_MAP = np.array([[0, 0, 0], [0, 255, 255]], dtype=np.uint8)


# ------------- 工具函数 -------------
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def to_tensor_nchw(np_img):
    # (H, W, C)->(1,C,H,W) float32
    x = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0).contiguous()
    return x

def norm_imagenet(np_img01):
    # np_img01: [0,1] float
    return (np_img01 - IMAGENET_MEAN) / IMAGENET_STD

def make_bool_mask_half(num_patches):
    # 上半不 mask，下半 mask（与训练/推理一致）
    m = torch.zeros(num_patches, dtype=torch.bool)
    m[num_patches // 2:] = True
    return m.unsqueeze(0)

def grayscale_mask_to_rgb01(mask_img, fg_value=255):
    """
    将单通道灰度掩码(0/255)转成3通道 [0,1]，前景=1，背景=0
    """
    if mask_img.mode != "L":
        mask_img = mask_img.convert("L")
    m = np.array(mask_img, dtype=np.uint8)
    m_bin = (m > 127).astype(np.float32)  # 0/1
    m3 = np.stack([m_bin, m_bin, m_bin], axis=-1)  # (H, W, 3)
    return m3  # [0,1]

# ----------------- 关键：索引解析 -----------------
def resolve_intermediate_indices(inter_cfg, keep_indices, num_layers_pruned):
    """
    inter_cfg: config.json 里的 intermediate_hidden_state_indices
    keep_indices: keep_indices.json 里的“原空间”层号（升序）
    num_layers_pruned: 裁后层数，一般==len(keep_indices)

    返回: (pruned_space_indices:list[int], mode:'pruned'或'orig')
    - 若 inter_cfg 已经是裁后 0..(L-1) 的下标（并且不是 keep_indices 本身），直接透传
    - 否则把 inter_cfg 当作原空间层号，映射到裁后空间
    """
    inter = [int(i) for i in inter_cfg]
    keep_set = set(keep_indices)

    # 情况1：看起来已经是裁后空间下标 -> 直接透传
    if all(0 <= i < num_layers_pruned for i in inter) and not set(inter).issubset(keep_set):
        return inter, "pruned"

    # 情况2：把 inter 当作原空间层号，映射到裁后空间
    pos_in_keep = {orig_idx: new_i for new_i, orig_idx in enumerate(keep_indices)}

    def nearest_kept(orig_idx):
        return min(keep_indices, key=lambda k: abs(k - orig_idx))

    result = []
    used = set()
    for oi in inter:
        if oi in pos_in_keep:
            cand = pos_in_keep[oi]
        else:
            cand = pos_in_keep[nearest_kept(oi)]
        if cand in used:
            unused = [k for k in keep_indices if pos_in_keep[k] not in used]
            if unused:
                cand = pos_in_keep[min(unused, key=lambda k: abs(k - oi))]
        used.add(cand)
        result.append(cand)

    return result, "orig"


# ----------------- 构建模型（基于裁剪配置） -----------------
from functools import partial
from SegGPT.SegGPT_inference.models_seggpt import SegGPT  # ⬅️ 直接引入类

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

    # 解析中间层索引（仅当 inter_cfg 是“原空间层号”时才映射）
    inter_after, mode = resolve_intermediate_indices(inter_cfg, keep, num_layers)
    print(f"[pruned-cfg] keep_indices(original-space) = {keep}")
    print(f"[pruned-cfg] intermediate indices mode={mode}; {inter_cfg} -> {inter_after}")

    # 直接实例化 SegGPT —— 不走工厂函数，避免“multiple values”冲突
    model = SegGPT(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=cfg.get("num_channels", 3),
        embed_dim=embed_dim,
        depth=num_layers,                 # ⬅️ 用裁剪后的层数
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop_path_rate=drop_path,
        norm_layer=partial(torch.nn.LayerNorm, eps=cfg.get("layer_norm_eps", 1e-6)),
        act_layer=torch.nn.GELU,
        use_abs_pos=True,
        use_rel_pos=use_rel,
        rel_pos_zero_init=True,
        window_size=14,                   # 可保留；但为了安全，这里不指定 window_block_indexes
        window_block_indexes=(),          # ⬅️ 裁剪后索引映射复杂，干脆禁用 window attention（稳妥）
        residual_block_indexes=[],
        use_act_checkpoint=False,
        pretrain_img_size=pre_imsize,
        pretrain_use_cls_token=True,
        out_feature="last_feat",
        decoder_embed_dim=decoder_dim,    # ⬅️ 与 config 对齐
        loss_func="smoothl1",
    ).to(device).eval()

    # 告诉模型要收集哪些中间层（如果你的实现读取这个属性）
    setattr(model, "intermediate_hidden_state_indices", tuple(inter_after))

    # 若你的 forward_encoder 内部还写死了 [5,11,17,23]，做一次轻量 monkey patch（与之前相同）
    if not hasattr(model, "_forward_encoder_patched"):
        try:
            orig_fenc = model.forward_encoder
            def patched_forward_encoder(imgs, tgts, bool_masked_pos, seg_type, merge_between_batch=-1):
                x = model.patch_embed(imgs)
                y = model.patch_embed(tgts)
                B, Hp, Wp, _ = x.size()

                mask_token = model.mask_token.expand(B, Hp, Wp, -1)
                w = bool_masked_pos.unsqueeze(-1).type_as(mask_token).reshape(-1, Hp, Wp, 1)
                y = y * (1 - w) + mask_token * w

                x = x + model.segment_token_x
                y = y + model.segment_token_y

                if getattr(model, "pos_embed", None) is not None:
                    from SegGPT.SegGPT_inference.util.vitdet_utils import get_abs_pos
                    x = x + get_abs_pos(model.pos_embed, model.pretrain_use_cls_token, (x.shape[1], x.shape[2]))
                    y = y + get_abs_pos(model.pos_embed, model.pretrain_use_cls_token, (y.shape[1], y.shape[2]))

                type_emb = torch.zeros(B, 1, 1, model.type_token_cls.shape[-1]).to(x.device)
                type_emb[seg_type == 0] = model.type_token_cls
                type_emb[seg_type == 1] = model.type_token_ins

                x = x + type_emb
                y = y + type_emb
                x = torch.cat((x, y), dim=0)
                merge_idx = 2

                outs = []
                for idx, blk in enumerate(model.blocks):
                    merge = 0
                    if merge_between_batch >= 0 and idx >= merge_between_batch:
                        merge = 1 if merge_idx >= idx else 2
                    x = blk(x, merge=merge)
                    if idx == merge_idx:
                        x = (x[:x.shape[0] // 2] + x[x.shape[0] // 2:]) * 0.5
                    if idx in inter_after:
                        outs.append(model.norm(x))
                return outs

            model.forward_encoder = patched_forward_encoder
            model._forward_encoder_patched = True
        except Exception:
            pass

    # 最后再做一次 decoder 输入维度的 sanity check
    expect_in = embed_dim * len(inter_after)
    if hasattr(model, "decoder_embed") and isinstance(model.decoder_embed, torch.nn.Linear):
        if model.decoder_embed.in_features != expect_in:
            raise RuntimeError(
                f"decoder_embed.in_features={model.decoder_embed.in_features} "
                f"but expect {expect_in} (=embed_dim*#intermediate_layers). "
                f"请确保构造参数/权重与中间层个数一致。"
            )

    return model

# ----------------- 加载微调 .pt -----------------
def load_finetuned_pt(model, ckpt_path, device):
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict):
        if "model_state_dict" in obj:
            sd = obj["model_state_dict"]
        elif "state_dict" in obj:
            sd = obj["state_dict"]
        else:
            # 可能就是 state_dict
            sd = obj
    else:
        sd = obj

    # 只加载匹配形状的键
    msd = model.state_dict()
    filtered = {k: v for k, v in sd.items() if (k in msd and msd[k].shape == v.shape)}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"[load finetuned pt] loaded={len(filtered)} / model_params={len(msd)}")
    if missing:
        print(f"[load finetuned pt] missing={len(missing)}  e.g. {missing[:10]}")
    if unexpected:
        print(f"[load finetuned pt] unexpected={len(unexpected)}  e.g. {unexpected[:10]}")
    model.to(device).eval()
    return model


# ----------------- 单 prompt 推理 -----------------
@torch.no_grad()
def run_one_image_single_prompt(model, device, image_path, prompt_image_path, prompt_mask_path, out_path):
    # 读图并缩放到 448x448（你的 pruned config 里就是 896x448 的双图拼接；单图各 448）
    res, hres = 448, 448

    # input image
    im = Image.open(image_path).convert("RGB").resize((res, hres))
    img_np01 = np.asarray(im, dtype=np.float32) / 255.0

    # prompt image
    pim = Image.open(prompt_image_path).convert("RGB").resize((res, hres))
    p_np01 = np.asarray(pim, dtype=np.float32) / 255.0

    # prompt mask -> 3通道 0/1
    pm = Image.open(prompt_mask_path).convert("L").resize((res, hres), Image.NEAREST)
    pm_rgb01 = grayscale_mask_to_rgb01(pm).astype(np.float32)

    # 构造“输入拼接”和“目标拼接”（与训练一致：上半 prompt，下半输入）
    # 注意：这里我们把 prompt 的 target 直接用 pm_rgb01（0/1 三通道）
    tgt2 = pm_rgb01
    img2 = p_np01

    # 目标 = [prompt_mask(三通道), 待预测图像占位]；这里为了匹配训练 scale，
    # 我们把“待预测图像占位”也用 prompt_mask 的三通道（不会被用来监督，后续会 mask 掉）
    tgt = np.concatenate([tgt2, tgt2], axis=0)     # (2*H, W, 3)
    img = np.concatenate([img2, img_np01], axis=0) # (2*H, W, 3)

    # 归一化
    img = norm_imagenet(img)
    tgt = norm_imagenet(tgt)

    # -> tensor
    x = to_tensor_nchw(img)   # (1,3,2H,W)
    y = to_tensor_nchw(tgt)   # (1,3,2H,W)

    # bool_masked_pos：上半 0（keep），下半 1（remove）
    # num_patches = (2H/16)*(W/16) = 2 * 28 * 28 = 1568
    H, W = 2 * hres, res
    num_patches = (H // 16) * (W // 16)
    bool_masked_pos = make_bool_mask_half(num_patches)

    valid = torch.ones_like(y)
    seg_type = torch.zeros([1, 1])  # instance=1 / semantic=0，按你的训练设定，默认 0
    feat_ensemble = -1  # 单prompt，-1 等价不做跨 batch ensemble

    x = x.to(device)
    y = y.to(device)
    bool_masked_pos = bool_masked_pos.to(device)
    valid = valid.to(device)
    seg_type = seg_type.to(device)

    # 前向
    loss, pred_patchified, _ = model(
        x.float(), y.float(), bool_masked_pos, valid.float(), seg_type, feat_ensemble
    )
    pred = model.unpatchify(pred_patchified)           # (1,3,2H,W)
    pred = pred.permute(0, 2, 3, 1).detach().cpu()     # (1,2H,W,3)

    # 取下半部分（待预测图像对应的下半）
    pred_bottom = pred[:, pred.shape[1] // 2 :, :, :]  # (1,H,W,3)
    pred_bottom = torch.clip((pred_bottom * IMAGENET_STD + IMAGENET_MEAN) * 255.0, 0, 255)

    # 简单二类：阈值到0/1，然后上色
    # 这里取单通道阈值（也可取三通道平均）
    gray = pred_bottom[0].mean(dim=-1).numpy()  # (H,W)
    bin_mask = (gray >= 128).astype(np.uint8)   # 0/1
    color = BIN_COLOR_MAP[bin_mask]             # (H,W,3) uint8

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(color).save(out_path)
    print(f"[inference] saved to: {out_path}")


# ----------------- CLI -----------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="微调得到的 .pt (logs/.../weights/*.pt)")
    p.add_argument("--pruned-dir", type=str, required=True, help="包含 config.json / keep_indices.json 的目录")
    p.add_argument("--image", type=str, required=True, help="待分割图")
    p.add_argument("--prompt-image", type=str, required=True, help="单张 prompt 图")
    p.add_argument("--prompt-mask", type=str, required=True, help="prompt 的灰度掩码(0/255)")
    p.add_argument("--out", type=str, required=True, help="输出路径，如 outputs/pred.png")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1) 基于裁剪配置构建模型（深度、相对位置、decoder维度、intermediate层号）
    model = build_model_from_pruned_cfg(args.pruned_dir, device)

    # 2) 加载微调 .pt
    model = load_finetuned_pt(model, args.ckpt, device)

    # 3) 单 prompt 推理
    with torch.no_grad():
        run_one_image_single_prompt(
            model, device,
            image_path=args.image,
            prompt_image_path=args.prompt_image,
            prompt_mask_path=args.prompt_mask,
            out_path=args.out,
        )


if __name__ == "__main__":
    main()

'''
python infer_with_pruned_cfg.py \
  --ckpt logs/1756021706/weights/best.pt \
  --pruned-dir pruned/pruned_seggpt_50 \
  --image data/train/images/shui1_0.png \
  --prompt-image data/save4_2.jpeg \
  --prompt-mask  data/save4_2_mask.jpeg \
  --out outputs/pred.png \
  --device cuda
'''