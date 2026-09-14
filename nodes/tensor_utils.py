"""Helpers tensor partages par les nodes d'aioli-nodes."""

import numpy as np
import torch
from PIL import Image


def resize_lanczos(tensor, new_w, new_h, mode):
    """
    Resize Lanczos d'un batch ComfyUI.
      mode="image" : tensor (B, H, W, C)
      mode="mask"  : tensor (B, H, W)
    """
    frames = []
    for b in range(tensor.shape[0]):
        arr = (tensor[b].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        if mode == "image":
            pil = Image.fromarray(arr).resize((new_w, new_h), Image.LANCZOS)
        else:
            pil = Image.fromarray(arr, mode="L").resize((new_w, new_h), Image.LANCZOS)
        frames.append(np.array(pil).astype(np.float32) / 255.0)
    return torch.from_numpy(np.stack(frames))
