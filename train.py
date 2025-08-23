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

def map_hf_key_to_custom(hf_key: str) -> Optional[str]:
    """
    把 HF Transformers 的 key，映射到你自定义实现的 key。
    返回 None 表示此权重在自定义实现中无对应（跳过）。
    """
    k = hf_key

    # 1) 去掉顶层前缀
    k = strip_prefix(k, "model.")          # model.encoder... -> encoder...
    # 2) encoder -> 顶层
    k = strip_prefix(k, "encoder.")        # encoder.blocks.0.* -> blocks.0.*

    # 3) embeddings 区
    k = k.replace("embeddings.patch_embeddings.projection.", "patch_embed.proj.")
    k = k.replace("embeddings.position_embeddings", "pos_embed")
    k = k.replace("embeddings.mask_token", "mask_token")
    # 下面三个是你自定义里的 token 命名；若 HF 有对应键就能映射
    k = k.replace("embeddings.segment_token_x", "segment_token_x")
    k = k.replace("embeddings.segment_token_y", "segment_token_y")
    k = k.replace("embeddings.type_token_cls", "type_token_cls")
    k = k.replace("embeddings.type_token_ins", "type_token_ins")

    # 4) blocks 内部（基本一致）
    # ... attn.qkv/proj, rel_pos_h/w, mlp.fc1/fc2, norm1/norm2 一般名字都相同

    # 5) decoder 区（HF: decoder.decoder_* ; 你：decoder_embed + decoder_pred Sequential[0,1,2,3]）
    if k.startswith("decoder."):
        k = k.replace("decoder.decoder_embed.", "decoder_embed.")
        # conv / layernorm / head -> 顺序容器 0/1/3
        k = k.replace("decoder.decoder_pred.conv.", "decoder_pred.0.")
        k = k.replace("decoder.decoder_pred.layernorm.", "decoder_pred.1.")
        k = k.replace("decoder.decoder_pred.head.", "decoder_pred.3.")
        # 其它像 gelu 在 HF 里没有权重，不用管

    # 6) 可能还有 "layer" 命名差异（少见）：encoder.layers -> blocks
    k = k.replace("layers.", "blocks.")

    return k

def build_custom_from_hf_config(cfg: dict) -> SegGPT:
    img_size   = tuple(cfg.get("image_size", [896, 448]))
    patch_size = cfg.get("patch_size", 16)
    embed_dim  = cfg.get("hidden_size", 1024)
    depth      = cfg.get("num_hidden_layers", 24)
    nheads     = cfg.get("num_attention_heads", 16)
    drop_path  = cfg.get("drop_path_rate", 0.0)
    use_rel    = cfg.get("use_relative_position_embeddings", True)
    pre_imsize = cfg.get("pretrain_image_size", 224)
    dec_dim    = cfg.get("decoder_hidden_size", 64)
    inter_idx  = cfg.get("intermediate_hidden_state_indices", [5,11,17,23])

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
        pretrain_use_cls_token=True,   # ✅ 打开 CLS，对应 HF 的 197 长度
        out_feature="last_feat",
        decoder_embed_dim=dec_dim,
        loss_func="smoothl1",
        intermediate_hidden_state_indices=tuple(inter_idx),
    )
    return model


def load_hf_safetensors_into_custom(ckpt_dir: str, device="cpu") -> SegGPT:
    # 1) 读 HF config
    cfg = json.load(open(os.path.join(ckpt_dir, "config.json"), "r"))
    # 2) 构建自定义模型（结构对齐）
    model = build_custom_from_hf_config(cfg)
    model = model.to(device)

    # 3) 载入 HF 权重
    hf_sd = safe_load(os.path.join(ckpt_dir, "model.safetensors"), device="cpu")

    # 4) 按映射规则生成“自定义命名”的 state_dict
    mapped = {}
    miss    = []
    for k, v in hf_sd.items():
        newk = map_hf_key_to_custom(k)
        if newk is None:
            miss.append(k); continue
        mapped[newk] = v

    # 4.x) pos_embed 自适配（必须在过滤前）
    model_sd = model.state_dict()  # 提前拿到目标形状
    if "pos_embed" in mapped and "pos_embed" in model_sd:
        pe = mapped["pos_embed"]
        need = model_sd["pos_embed"].shape[1]
        have = pe.shape[1]
        if have != need and abs(have - need) == 1:
            if have > need:
                # 多了 CLS：去掉第一个 token
                mapped["pos_embed"] = pe[:, 1:, :]
            else:
                # 少了 CLS：在最前面补一个全零 token
                pad = torch.zeros(pe.shape[0], 1, pe.shape[2], dtype=pe.dtype)
                mapped["pos_embed"] = torch.cat([pad, pe], dim=1)
    # 5) 只保留模型里存在的键（避免 shape/key 冲突）
    model_sd = model.state_dict()
    filtered = {}
    skipped_shape = []
    for k, v in mapped.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            filtered[k] = v
        else:
            if k not in model_sd:
                miss.append(k)
            else:
                skipped_shape.append((k, tuple(v.shape), tuple(model_sd[k].shape)))

    # 6) load（strict=False），并打印命中统计
    missing, unexpected = model.load_state_dict(filtered, strict=False)

    hit = len(filtered)
    total = len(model_sd)
    print(f"[HF->Custom] loaded={hit}/{total} params")
    if skipped_shape:
        print("[HF->Custom] shape-mismatch keys (skip):")
        for k, hs, ms in skipped_shape[:20]:
            print(f"  - {k}: hf{hs} vs model{ms}")
        if len(skipped_shape) > 20:
            print(f"  ... and {len(skipped_shape)-20} more")
    if missing:
        print(f"[HF->Custom] missing in provided dict: {len(missing)} (model expected but not provided)")
        print("  e.g.", missing[:10])
    if unexpected:
        print(f"[HF->Custom] unexpected: {len(unexpected)} (provided but model unused)")
        print("  e.g.", unexpected[:10])

    # 7) 额外一致性检查（decoder 输入维是否与 indices 个数一致）
    if isinstance(model.decoder_embed, nn.Linear):
        embed_dim = model._out_feature_channels[model._out_features[0]]
        expect_in = embed_dim * len(cfg.get("intermediate_hidden_state_indices", [5,11,17,23]))
        if model.decoder_embed.in_features != expect_in:
            raise RuntimeError(
                f"decoder_embed.in_features={model.decoder_embed.in_features} "
                f"but expect {expect_in} (=embed_dim*#indices). "
                f"请确认 SegGPT 构造时已按 indices 个数设置 decoder 输入维。"
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
