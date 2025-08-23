import os
import json
from typing import Iterable, List, Dict, Any, Optional

import torch
import torch.nn as nn

# ----------------- helpers -----------------

def _as_module_list(x):
    # 有些模型把 blocks 存成 list，不是 ModuleList；统一转一下
    return x if isinstance(x, nn.ModuleList) else nn.ModuleList(list(x))

def find_encoder_blocks(model: nn.Module):
    """Return (encoder_module, attribute_name, blocks: ModuleList)."""
    candidates = [getattr(model, 'seggpt', None), getattr(model, 'model', None), model]
    for root in candidates:
        if root is None:
            continue
        encoder = getattr(root, 'encoder', root)
        for name in ['blocks', 'layers', 'layer']:
            blocks = getattr(encoder, name, None)
            if isinstance(blocks, (nn.ModuleList, list, tuple)) and len(blocks) > 0:
                return encoder, name, _as_module_list(blocks)
    raise RuntimeError('Cannot locate encoder blocks ModuleList')
def _normalize_keep_indices(raw: Any, num_blocks: int) -> List[int]:
    """
    接受多种格式：
    - [0, 2, 4]  或  ["0","2","4"]
    - {"keep_indices":[...]} / {"kept":[...]} / {"indices":[...]} / {"blocks":[...]} / {"keep":[...]}
    - {"mask":[true,false,true,...]} 或 ["1","0","1",...]
    - "0,2,4"
    自动：
    - 转 int
    - 去重、排序
    - 必要时把 1-based 索引转为 0-based（启发式判断）
    - 越界校验
    - 兼容：若提供的是“原始空间的保留列表”且长度==当前层数，但含有越界值，则视作“保留当前全部层”，映射为 range(num_blocks)
    """
    # 1) dict：常见 key
    if isinstance(raw, dict):
        for k in ['keep_indices', 'kept', 'indices', 'blocks', 'keep']:
            if k in raw:
                raw = raw[k]
                break
        else:
            # 掩码写法
            for k in ['mask', 'keep_mask', 'kept_mask']:
                if k in raw:
                    mask = raw[k]
                    idx = [i for i, v in enumerate(mask) if (v is True or v == 1 or str(v).lower() == "1")]
                    return idx
            raise ValueError("Unrecognized dict structure in keep_indices.json")

    # 2) 字符串：逗号分隔或单个
    if isinstance(raw, str):
        raw = raw.replace(' ', '')
        if ',' in raw:
            raw = raw.split(',')
        else:
            raw = [raw]

    # 3) list/tuple：可能是掩码或索引
    if isinstance(raw, (list, tuple)):
        # 掩码（全为布尔/01 且长度匹配当前层数）
        as_str = [str(v).lower() for v in raw]
        if all(v in ['true', 'false', '0', '1'] for v in as_str) and len(raw) == num_blocks:
            idx = [i for i, v in enumerate(as_str) if v in ['true', '1']]
            return idx

        # 普通索引：转 int、去重、排序
        try:
            idx = sorted({int(x) for x in raw})
        except Exception as e:
            raise TypeError(f"keep_indices contains non-integer values: {raw}") from e

        # 1-based → 0-based 启发式
        if len(idx) > 0 and 0 not in idx and idx[0] >= 1 and (num_blocks is not None) and idx[-1] <= num_blocks:
            idx = [i - 1 for i in idx]

        # --- 兼容：原始空间的保留列表（长度等于当前层数但含越界）
        if num_blocks is not None and len(idx) == num_blocks and (len(idx) > 0 and max(idx) >= num_blocks):
            remapped = list(range(num_blocks))
            print(f"[keep_indices] Detected original-space keep list; remap -> {remapped}")
            return remapped

        # 越界/负数检查
        bad = [i for i in idx if i < 0 or (num_blocks is not None and i >= num_blocks)]
        if bad:
            raise IndexError(f"keep_indices out of range (num_blocks={num_blocks}): {bad}")

        return idx

    raise TypeError(f"Unsupported keep_indices type: {type(raw)}")


# ----------------- core funcs -----------------

def prune_model(model: nn.Module, keep_indices: Iterable[int]):
    """Prune transformer blocks according to keep_indices (0-based)."""
    encoder, attr, blocks = find_encoder_blocks(model)
    keep_indices = list(keep_indices)
    keep_blocks = [blocks[i] for i in keep_indices]  # 这里要求 int 索引
    setattr(encoder, attr, nn.ModuleList(keep_blocks))
    # update config if available
    if hasattr(model, 'config') and hasattr(model.config, 'num_hidden_layers'):
        model.config.num_hidden_layers = len(keep_blocks)
    return model


# pruning_utils.py

import os, json, torch
from typing import Any, List, Dict
import os, json, torch

from hf_keymap import map_hf_key_to_custom, looks_like_hf

def load_pruned_checkpoint(model, pruned_dir: str):
    """
    支持 pytorch_model.bin / model.safetensors；支持 keep_indices.json；
    若权重看起来是 HF 命名，自动键名映射与 pos_embed 兜底。
    """
    bin_path = os.path.join(pruned_dir, "pytorch_model.bin")
    safetensors_path = os.path.join(pruned_dir, "model.safetensors")

    # 读取 state_dict
    if os.path.isfile(safetensors_path):
        from safetensors.torch import load_file as load_safetensors
        state_dict = load_safetensors(safetensors_path)
    elif os.path.isfile(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No checkpoint found in '{pruned_dir}'.")

    # 先处理 keep_indices：先剪枝，再加载
    keep_path = os.path.join(pruned_dir, "keep_indices.json")
    if os.path.exists(keep_path):
        with open(keep_path, "r") as f:
            raw = json.load(f)
        # 你已有 _normalize_keep_indices / prune_model
        num_blocks = len(getattr(model, "blocks"))
        keep_indices = _normalize_keep_indices(raw, num_blocks=num_blocks)
        model = prune_model(model, keep_indices)

    # 如果是 HF 命名：做键名映射 + pos_embed 兜底
    if looks_like_hf(state_dict.keys()):
        mapped = {}
        for k, v in state_dict.items():
            nk = map_hf_key_to_custom(k)
            if nk is not None:
                mapped[nk] = v
        # pos_embed 长度差 1 的兜底
        msd = model.state_dict()
        if "pos_embed" in mapped and "pos_embed" in msd:
            pe = mapped["pos_embed"]
            need = msd["pos_embed"].shape[1]
            have = pe.shape[1]
            if have != need and abs(have - need) == 1:
                if have > need:
                    mapped["pos_embed"] = pe[:, 1:, :]  # 去掉 CLS
                else:
                    pad = torch.zeros(pe.shape[0], 1, pe.shape[2], dtype=pe.dtype)
                    mapped["pos_embed"] = torch.cat([pad, pe], dim=1)
        state_dict = mapped

    # 仅加载模型中存在且形状一致的键
    msd = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if (k in msd and msd[k].shape == v.shape)}
    missing, unexpected = model.load_state_dict(filtered, strict=False)

    # 用 mask_token 初始化四个额外 token（HF 通常没有）
    with torch.no_grad():
        if hasattr(model, "mask_token"):
            for t in ["segment_token_x", "segment_token_y", "type_token_cls", "type_token_ins"]:
                if hasattr(model, t):
                    param_t = getattr(model, t)
                    # 若仍是零（默认初始化），用 mask_token 赋值
                    if torch.count_nonzero(param_t) == 0:
                        param_t.data.copy_(model.mask_token.data)

    # 打印简要统计
    print(f"[load_pruned_checkpoint] loaded={len(filtered)} / model_params={len(msd)}")
    if missing:
        print(f"[load_pruned_checkpoint] missing: {len(missing)}  e.g. {missing[:10]}")
    if unexpected:
        print(f"[load_pruned_checkpoint] unexpected: {len(unexpected)}  e.g. {unexpected[:10]}")
    return model
