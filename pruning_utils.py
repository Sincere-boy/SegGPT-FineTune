import os
import json
from typing import Iterable

import torch
import torch.nn as nn


def find_encoder_blocks(model: nn.Module):
    """Return encoder module, attribute name and blocks ModuleList.

    This walks through common attribute names used by SegGPT models
    to locate the list of transformer blocks.
    """
    candidates = [getattr(model, 'seggpt', None), getattr(model, 'model', None), model]
    for root in candidates:
        if root is None:
            continue
        encoder = getattr(root, 'encoder', root)
        for name in ['blocks', 'layers', 'layer']:
            blocks = getattr(encoder, name, None)
            if isinstance(blocks, nn.ModuleList):
                return encoder, name, blocks
    raise RuntimeError('Cannot locate encoder blocks ModuleList')


def prune_model(model: nn.Module, keep_indices: Iterable[int]):
    """Prune transformer blocks according to ``keep_indices``.

    ``keep_indices`` should be an iterable of block indices to retain in
    the original order.  The function modifies ``model`` in-place and
    returns it for convenience.
    """
    encoder, attr, blocks = find_encoder_blocks(model)
    keep_blocks = [blocks[i] for i in keep_indices]
    setattr(encoder, attr, nn.ModuleList(keep_blocks))
    # update config if available
    if hasattr(model, 'config') and hasattr(model.config, 'num_hidden_layers'):
        model.config.num_hidden_layers = len(keep_blocks)
    return model


def load_pruned_checkpoint(model: nn.Module, pruned_dir: str):
    """Load pruned weights into ``model``.

    The directory is expected to contain ``pytorch_model.bin`` with the
    state dict and an optional ``keep_indices.json`` recording the kept
    block indices.  If the JSON file exists, the model is pruned before
    the state dict is loaded.
    """
    ckpt_path = os.path.join(pruned_dir, 'pytorch_model.bin')
    state_dict = torch.load(ckpt_path, map_location='cpu')
    keep_path = os.path.join(pruned_dir, 'keep_indices.json')
    if os.path.exists(keep_path):
        with open(keep_path, 'r') as f:
            keep_indices = json.load(f)
        model = prune_model(model, keep_indices)
    model.load_state_dict(state_dict, strict=False)
    return model
