"""
Coeur du 🎞️ Aioli Video Compare (fonctions pures, testables hors ComfyUI).

Lecture : sequence d'images sur <canvas>, synchro a l'image pres.
  - proxys JPEG (cote long <= PROXY_SIDE, qualite PROXY_QUALITY) pour la
    lecture, ecrits dans le dossier temp de ComfyUI et servis par /view ;
  - image pleine resolution SANS PERTE (PNG) a la demande, pour l'image
    courante (pause, pas a pas, zoom 1:1, difference).
Pourquoi pas une <video> encodee (PyAV) : le 4:2:0 et la compression lissent
precisement le detail qu'on veut juger (upscale), le mode difference montrerait
le bruit d'encodage, et deux decodeurs video ne restent pas cales a l'image pres.
"""

import io
import os
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import numpy as np
import torch
from PIL import Image

from .tensor_utils import to_uint8

PROXY_SIDE = 1280
PROXY_QUALITY = 90


def frames_of(value, default_fps):
    """
    IMAGE (B, H, W, C) ou VIDEO (comfy_api VideoInput) -> (frames, fps).
    frames : tensor (N, H, W, 3) float 0-1 ; fps : float.
    """
    if isinstance(value, torch.Tensor):
        frames, fps = value, float(default_fps)
    else:
        comp = value.get_components()
        frames = comp.images
        rate = comp.frame_rate
        fps = float(rate) if isinstance(rate, (Fraction, int, float)) else float(default_fps)
    if frames.dim() == 3:
        frames = frames.unsqueeze(0)
    if frames.shape[-1] == 1:
        frames = frames.repeat(1, 1, 1, 3)
    elif frames.shape[-1] > 3:
        frames = frames[..., :3]
    if fps <= 0:
        fps = float(default_fps)
    return frames, fps


def proxy_size(w, h, side=PROXY_SIDE):
    """Taille du proxy : cote long <= side, ratio garde, pas d'agrandissement."""
    s = min(1.0, side / max(w, h))
    return max(1, round(w * s)), max(1, round(h * s))


def frame_uint8(frames, i, size=None):
    """Image i en uint8 (H, W, 3), redimensionnee (Lanczos) si size != taille."""
    arr = to_uint8(frames[i].detach().cpu().float().numpy())
    if size is not None and (arr.shape[1], arr.shape[0]) != tuple(size):
        arr = np.array(Image.fromarray(arr).resize(tuple(size), Image.LANCZOS))
    return arr


def write_proxies(frames, folder, prefix, size, workers=4):
    """Ecrit les proxys JPEG prefix_000000.jpg ... dans folder. Renvoie le nombre."""
    os.makedirs(folder, exist_ok=True)

    def one(i):
        arr = frame_uint8(frames, i, size)
        Image.fromarray(arr).save(os.path.join(folder, f"{prefix}_{i:06d}.jpg"),
                                  quality=PROXY_QUALITY, subsampling=0)

    n = int(frames.shape[0])
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, range(n)))
    return n


def full_png(frames, i, size=None):
    """Image i pleine resolution en PNG sans perte (bytes)."""
    buf = io.BytesIO()
    Image.fromarray(frame_uint8(frames, i, size)).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def describe(frames, fps, label):
    n, h, w = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
    return dict(label=str(label), n=n, fps=fps, w=w, h=h, duration=n / fps)
