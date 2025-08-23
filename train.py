import sys
sys.path.append('SegGPT/SegGPT_inference')

import os
import argparse
import json
import torch as T
import torch.multiprocessing as mp
from typing import Optional
from typing import Union
from agent import Agent
from typing import Dict
from utils import *
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from SegGPT.SegGPT_inference.models_seggpt import seggpt_vit_large_patch16_input896x448
from pruning_utils import load_pruned_checkpoint
from data import BaseDataset
from hf_keymap import map_hf_key_to_custom



def ddp_setup(rank: int, world_size: int, port:int=None):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(np.random.randint(10000, 60000)) if port is None else str(port)
    T.cuda.set_device(rank)
    T.cuda.empty_cache()
    init_process_group('nccl', rank=rank, world_size=world_size)

import os, json, re
import torch
import torch.nn as nn
from safetensors.torch import load_file as safe_load
from SegGPT.SegGPT_inference.models_seggpt import SegGPT

# ---------- 小工具 ----------
def strip_prefix(k: str, prefix: str) -> str:
    return k[len(prefix):] if k.startswith(prefix) else k

# ===== train.py 里的工具函数：HF -> 自定义 =====
import os, json, re
from typing import Optional, Dict
import torch
import torch.nn as nn

# 若已有同名函数，可删掉这里
def strip_prefix(s: str, prefix: str) -> str:
    return s[len(prefix):] if s.startswith(prefix) else s

def safe_load(path: str, device="cpu") -> Dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file as load_safetensors
        return load_safetensors(path, device=device)
    else:
        return torch.load(path, map_location=device)

# --------- 键名映射（HF -> 自定义） ----------
import re
from typing import Optional

_LIDX = re.compile(r"^(?:model\.)?encoder\.(layers|layer|blocks)\.(\d+)\.(.*)$")

# --------- 按 HF config 构建自定义模型结构 ----------
def build_custom_from_hf_config(cfg: dict) -> nn.Module:
    img_size   = tuple(cfg.get("image_size", [896, 448]))
    patch_size = cfg.get("patch_size", 16)
    embed_dim  = cfg.get("hidden_size", 1024)
    depth      = cfg.get("num_hidden_layers", 24)
    nheads     = cfg.get("num_attention_heads", 16)
    drop_path  = cfg.get("drop_path_rate", 0.0)
    use_rel    = cfg.get("use_relative_position_embeddings", True)
    pre_imsize = cfg.get("pretrain_image_size", 224)
    dec_dim    = cfg.get("decoder_hidden_size", 64)
    inter_idx  = cfg.get("intermediate_hidden_state_indices", [5, 11, 17, 23])

    # 这里的 SegGPT 类就是你自定义实现（已存在于项目中）
    model = SegGPT(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=cfg.get("num_channels", 3),
        embed_dim=embed_dim,
        depth=depth,
        num_heads=nheads,
        mlp_ratio=cfg.get("mlp_ratio", 4.0),
        qkv_bias=cfg.get("qkv_bias", True),
        drop_path_rate=drop_path,
        norm_layer=lambda d: nn.LayerNorm(d, eps=cfg.get("layer_norm_eps", 1e-6)),
        act_layer=nn.GELU,
        use_abs_pos=True,
        use_rel_pos=use_rel,
        rel_pos_zero_init=True,
        window_size=14,
        window_block_indexes=(),
        residual_block_indexes=(),
        use_act_checkpoint=False,
        pretrain_img_size=pre_imsize,
        pretrain_use_cls_token=True,   # 与 HF 的 pos_embed(197) 对齐
        out_feature="last_feat",
        decoder_embed_dim=dec_dim,
        loss_func="smoothl1",
        intermediate_hidden_state_indices=tuple(inter_idx),
    )
    return model

# --------- 读取 HF safetensors -> 映射 -> 过滤 -> 加载 ----------
def load_hf_safetensors_into_custom(ckpt_dir: str, device="cpu") -> nn.Module:
    cfg_path = os.path.join(ckpt_dir, "config.json")
    hf_path  = os.path.join(ckpt_dir, "model.safetensors")

    cfg = json.load(open(cfg_path, "r"))
    model = build_custom_from_hf_config(cfg).to(device)
    hf_sd = safe_load(hf_path, device="cpu")

    # 映射键名
    mapped: Dict[str, torch.Tensor] = {}
    for k, v in hf_sd.items():
        newk = map_hf_key_to_custom(k)
        if newk is not None:
            mapped[newk] = v

    # pos_embed 兜底（CLS 长度差 1 的情况）
    model_sd = model.state_dict()
    if "pos_embed" in mapped and "pos_embed" in model_sd:
        pe = mapped["pos_embed"]
        need = model_sd["pos_embed"].shape[1]
        have = pe.shape[1]
        if have != need and abs(have - need) == 1:
            if have > need:  # 多一个 CLS
                mapped["pos_embed"] = pe[:, 1:, :]
            else:            # 少一个 CLS
                pad = torch.zeros(pe.shape[0], 1, pe.shape[2], dtype=pe.dtype)
                mapped["pos_embed"] = torch.cat([pad, pe], dim=1)

    # 过滤：仅加载模型里存在且形状一致的键
    filtered = {}
    for k, v in mapped.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            filtered[k] = v

    # 加载
    missing, unexpected = model.load_state_dict(filtered, strict=False)

    # 补齐四个 token：若 HF 无提供且当前为全零，用 mask_token 初始化
    with torch.no_grad():
        msd = model.state_dict()
        if "mask_token" in msd:
            for t in ["segment_token_x", "segment_token_y", "type_token_cls", "type_token_ins"]:
                if t in msd and torch.count_nonzero(msd[t]) == 0:
                    msd[t].copy_(msd["mask_token"])

    hit = len(filtered)
    total = len(model_sd)
    print(f"[HF->Custom] mapped={len(mapped)} loaded={hit} / model_params={total}")
    if missing:
        print(f"[HF->Custom] still-missing: {len(missing)}  e.g. {missing[:10]}")
    if unexpected:
        print(f"[HF->Custom] unexpected: {len(unexpected)}  e.g. {unexpected[:10]}")

    # 可选一致性检查（如果你的 decoder_embed 固定用4个中间层拼接）
    inter_idx = cfg.get("intermediate_hidden_state_indices", [5,11,17,23])
    if isinstance(model.decoder_embed, nn.Linear):
        embed_dim = model._out_feature_channels[model._out_features[0]]
        expect_in = embed_dim * len(inter_idx)
        if model.decoder_embed.in_features != expect_in:
            raise RuntimeError(
                f"decoder_embed.in_features={model.decoder_embed.in_features} "
                f"but expect {expect_in} (=embed_dim*#indices)."
            )

    return model


def main(rank: int, world_size: int, train_args: Dict, port: int):
    ddp_setup(rank, world_size, port)

    setup_logging()
    logger = get_logger(__name__, rank)

    logger.info('Preparing dataset')
    train_dataset = BaseDataset(
        root = train_args['train_dataset_dir'], 
        n_classes = train_args['n_classes'],
        mean = train_args['image_mean'],
        std = train_args['image_std'],
        mask_ratio = train_args['mask_ratio'],
        resize = (1024, 1024),
        is_train=True,
    )
    val_dataset = BaseDataset(
        root = train_args['val_dataset_dir'], 
        n_classes = train_args['n_classes'],
        mean = train_args['image_mean'],
        std = train_args['image_std'],
        mask_ratio = train_args['mask_ratio'],
        resize = (896, 448),
        is_train = False,
    )

    logger.info('Instantiating model and trainer agent')

    ckpt_dir = "/root/projects/SegGPT-FineTune/pruned/pruned_seggpt_50"  # 里面有 model.safetensors + config.json
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_hf_safetensors_into_custom(ckpt_dir, device=device)
    
    pruned_dir = train_args.get('pruned_dir')
    if pruned_dir:
        model = load_pruned_checkpoint(model, pruned_dir)
        logger.info(f'Loaded pruned checkpoint from {pruned_dir}')
    else:
        initial_ckpt = T.load('seggpt_vit_large.pth', map_location='cpu')
        model.load_state_dict(initial_ckpt['model'], strict=False)
        logger.info('Initial checkpoint loaded')

    trainer = Agent(model, rank, train_args)
    logger.info(f'Using {T.cuda.device_count()} GPU(s)')
    if 'model_path' in train_args:
        trainer.load_checkpoint(train_args['model_path'])

    logger.info('Instantiating dataloader')
    train_dataloader = T.utils.data.DataLoader(
        train_dataset,
        batch_size=train_args['batch_size'],
        shuffle=False,
        num_workers=train_args['num_workers'],
        pin_memory=True,
        sampler=DistributedSampler(train_dataset),
    )
    val_dataloader = T.utils.data.DataLoader(
        val_dataset,
        batch_size=train_args['batch_size'],
        shuffle=False,
        num_workers=train_args['num_workers'],
        pin_memory=True,
        sampler=DistributedSampler(val_dataset),
    )

    trainer.do_training(train_dataloader, val_dataloader, train_args['eval_per_epoch'])
    destroy_process_group()

def get_args_parser():
    parser = argparse.ArgumentParser('SegGPT train', add_help=False)
    parser.add_argument('--uid', type=str, help='unique id for the run', default=None)
    parser.add_argument('--port', type=int, help='DDP port', default=None)
    parser.add_argument('--config', type=str, help='path to json config', default='configs/base.json')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args_parser()
    train_args = json.load(open(args.config, 'r'))
    train_args['uid'] = args.uid
    world_size = T.cuda.device_count()
    mp.spawn(main, nprocs=world_size, args=(world_size, train_args, args.port))
