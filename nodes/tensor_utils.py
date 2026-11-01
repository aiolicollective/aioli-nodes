"""Helpers tensor partages par les nodes d'aioli-nodes."""

import numpy as np
import torch
from PIL import Image


def to_uint8(arr, rounding="nearest"):
    """
    float 0-1 -> uint8 pour PIL.
      nearest : arrondi au plus proche (v1.8). Sans biais.
      floor   : troncature (avant v1.8) : -0,5 niveau en moyenne sur des
                valeurs continues (sortie VAE, Color Fix). Expression exacte
                de l'ancien code (dtype d'origine) : les nodes legacy rendent
                a l'identique au bit pres.
    Une IMAGE LoadImage (k/255) donne le meme resultat dans les deux modes.
    Pas un MASK LoadImage (1 - k/255) : jusqu'a 1 niveau d'ecart.
    """
    if rounding == "floor":
        return (arr * 255).clip(0, 255).astype(np.uint8)
    if rounding != "nearest":
        raise ValueError(f"to_uint8 : rounding inconnu {rounding!r}")
    return np.rint(np.asarray(arr, dtype=np.float32) * 255.0).clip(0, 255).astype(np.uint8)


def resize_lanczos(tensor, new_w, new_h, mode, rounding="nearest"):
    """
    Resize Lanczos d'un batch ComfyUI.
      mode="image" : tensor (B, H, W, C)
      mode="mask"  : tensor (B, H, W)
    """
    frames = []
    for b in range(tensor.shape[0]):
        arr = to_uint8(tensor[b].cpu().numpy(), rounding)
        if mode == "image":
            pil = Image.fromarray(arr).resize((new_w, new_h), Image.LANCZOS)
        else:
            pil = Image.fromarray(arr, mode="L").resize((new_w, new_h), Image.LANCZOS)
        frames.append(np.array(pil).astype(np.float32) / 255.0)
    return torch.from_numpy(np.stack(frames))
