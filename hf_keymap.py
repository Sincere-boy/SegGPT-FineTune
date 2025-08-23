# hf_keymap.py
import re
from typing import Optional, Iterable

# 形如：model.encoder.layers.12.xxx / encoder.layers.0.xxx / model.encoder.blocks.0.xxx
_LIDX = re.compile(r"^(?:model\.)?encoder\.(layers|layer|blocks)\.(\d+)\.(.*)$")

def looks_like_hf(sd_keys: Iterable[str]) -> bool:
    """粗判 state_dict 的命名是否像 HF SegGpt。"""
    for k in sd_keys:
        if str(k).startswith(("model.", "encoder.", "embeddings.", "decoder.")):
            return True
        break
    return False

def map_hf_key_to_custom(hf_key: str) -> Optional[str]:
    """
    将 HuggingFace Transformers 的 SegGpt 权重 key 映射为你自定义实现中的 key。
    识别的 HF 命名：model.embeddings/*, model.encoder.layers/*, model.encoder.layernorm.*,
                   decoder.decoder_embed.*, decoder.decoder_pred.(conv|layernorm|head).*
    也兼容去掉顶层 'model.' 前缀后的 'embeddings./encoder./decoder.' 形式。
    返回 None 表示此 key 在自定义结构中无对应（跳过）。
    """

    k = hf_key

    # ---- 允许缺省 'model.' 前缀 ----
    if k.startswith("model."):
        k = k[len("model."):]

    # =======================
    # 1) Embeddings 区域
    # =======================
    if k.startswith("embeddings."):
        tail = k[len("embeddings."):]  # e.g. patch_embeddings.projection.weight
        # patch conv -> PatchEmbed.proj
        if tail.startswith("patch_embeddings.projection."):
            return "patch_embed.proj." + tail.split(".", 2)[-1]

        # 位置编码 -> pos_embed
        if tail == "position_embeddings":
            return "pos_embed"

        # mask token
        if tail == "mask_token":
            return "mask_token"

        # segment tokens（HF: input/prompt；自定义：x/y）
        if tail == "segment_token_input":
            return "segment_token_x"
        if tail == "segment_token_prompt":
            return "segment_token_y"

        # type tokens（HF: semantic/instance；自定义：cls/ins）
        if tail == "type_token_semantic":
            return "type_token_cls"
        if tail == "type_token_instance":
            return "type_token_ins"

        # 其它未识别的 embedding 键
        return None

    # =======================
    # 2) Encoder 顶层 LayerNorm
    # =======================
    if k.startswith("encoder.layernorm."):
        # encoder.layernorm.{weight,bias} -> norm.{weight,bias}
        return "norm." + k.split(".", 2)[-1]

    # =======================
    # 3) Encoder 各层（layers.N.* 或 layer.N.*）
    # =======================
    for layer_prefix in ("encoder.layers.", "encoder.layer."):
        if k.startswith(layer_prefix):
            tail = k[len(layer_prefix):]  # e.g. "0.attention.qkv.weight"
            parts = tail.split(".")
            if len(parts) < 2:
                return None
            try:
                layer_idx = int(parts[0])
            except ValueError:
                return None
            sub = ".".join(parts[1:])  # e.g. "attention.qkv.weight"
            prefix = f"blocks.{layer_idx}."

            # 3.1 attention
            if sub.startswith("attention."):
                t2 = sub[len("attention."):]  # qkv./proj./projection./rel_pos_h/rel_pos_w
                if t2.startswith("qkv."):
                    return prefix + "attn.qkv." + t2.split(".", 1)[1]
                # 兼容 proj. 与 projection.
                if t2.startswith("proj."):
                    return prefix + "attn.proj." + t2.split(".", 1)[1]
                if t2.startswith("projection."):
                    return prefix + "attn.proj." + t2.split(".", 1)[1]
                if t2 == "rel_pos_h":
                    return prefix + "attn.rel_pos_h"
                if t2 == "rel_pos_w":
                    return prefix + "attn.rel_pos_w"
                return None

            # 3.2 MLP（HF: lin1/lin2 -> 自定义: fc1/fc2）
            if sub.startswith("mlp."):
                t2 = sub[len("mlp."):]  # lin1.weight / lin2.bias
                if t2.startswith("lin1."):
                    return prefix + "mlp.fc1." + t2.split(".", 1)[1]
                if t2.startswith("lin2."):
                    return prefix + "mlp.fc2." + t2.split(".", 1)[1]
                return None

            # 3.3 LayerNorm（HF: layernorm_before/after -> 自定义: norm1/norm2）
            if sub.startswith("layernorm_before."):
                return prefix + "norm1." + sub.split(".", 1)[1]
            if sub.startswith("layernorm_after."):
                return prefix + "norm2." + sub.split(".", 1)[1]

            # 其它子模块没有对应
            return None

    # =======================
    # 4) Decoder（HF -> 自定义 Linear + Sequential）
    # =======================
    if k.startswith("decoder."):
        tail = k[len("decoder."):]  # decoder_embed.* / decoder_pred.*
        # 4.1 decoder_embed（Linear）
        if tail.startswith("decoder_embed."):
            return "decoder_embed." + tail.split(".", 1)[1]

        # 4.2 decoder_pred Head（conv/layernorm/head -> 顺序容器 0/1/3）
        if tail.startswith("decoder_pred."):
            t2 = tail[len("decoder_pred."):]

            if t2.startswith("conv."):
                return "decoder_pred.0." + t2.split(".", 1)[1]
            if t2.startswith("layernorm."):
                return "decoder_pred.1." + t2.split(".", 1)[1]
            if t2.startswith("head."):
                return "decoder_pred.3." + t2.split(".", 1)[1]
            # gelu 在 HF 里无权重，对应自定义 Sequential[2]，无需映射
            return None

        return None

    # 其它未识别前缀（可能是优化器状态、缓冲等），跳过
    return None
