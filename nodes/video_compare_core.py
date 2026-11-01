"""
Coeur du 🎞️ Aioli Video Compare (fonctions pures, testables hors ComfyUI).

Lecture : sequence d'images sur <canvas>, synchro a l'image pres. Pour chaque
clip, chaque image est ecrite UNE fois dans le dossier temp de ComfyUI :
  - <side>_<i>.jpg      proxy (cote long <= PROXY_SIDE, JPEG q90 4:4:4) : lecture ;
  - <side>_full_<i>.png pleine resolution SANS PERTE (PNG, compression rapide) :
                        pause, pas a pas, zoom, difference.
Rien n'est garde en memoire apres l'execution. Une VIDEO venant d'un fichier
est decodee en flux (PyAV, image par image) : jamais tout le clip en float32.

Pourquoi pas une <video> encodee : le 4:2:0 et la compression lissent le detail
qu'on veut juger (upscale), le mode difference montrerait le bruit d'encodage, et
deux decodeurs video ne restent pas cales a l'image pres.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import numpy as np
import torch
from PIL import Image

from .tensor_utils import to_uint8

PROXY_SIDE = 1280
PROXY_QUALITY = 90
PNG_LEVEL = 1


def proxy_size(w, h, side=PROXY_SIDE):
    """Taille du proxy : cote long <= side, ratio garde, pas d'agrandissement."""
    s = min(1.0, side / max(w, h))
    return max(1, round(w * s)), max(1, round(h * s))


def _as_fps(rate, default_fps):
    try:
        fps = float(rate)
    except (TypeError, ValueError):
        return float(default_fps)
    return fps if fps > 0 else float(default_fps)


def _rgb8(arr):
    """(H, W, C) uint8 -> (H, W, 3) uint8."""
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return np.ascontiguousarray(arr[..., :3])


def iter_frames(value, default_fps):
    """
    IMAGE (B, H, W, C) ou VIDEO (comfy_api VideoInput) -> (fps, iterateur d'images
    uint8 (H, W, 3)). VIDEO d'un fichier : decodage en flux par PyAV ; sinon
    get_components() (tout en memoire, comme le reste de ComfyUI).
    """
    if isinstance(value, torch.Tensor):
        t = value if value.dim() == 4 else value.unsqueeze(0)
        return float(default_fps), (_rgb8(to_uint8(t[i].detach().cpu().float().numpy()))
                                    for i in range(int(t.shape[0])))
    src = _file_source(value)
    if src is not None:
        import av
        if hasattr(src, "seek"):
            src.seek(0)
        container = av.open(src)
        stream = container.streams.video[0]
        fps = _as_fps(stream.average_rate or stream.guessed_rate, default_fps)

        def gen():
            try:
                for frame in container.decode(stream):
                    yield _rgb8(frame.to_ndarray(format="rgb24"))
            finally:
                container.close()
        return fps, gen()
    comp = value.get_components()
    images = comp.images if comp.images.dim() == 4 else comp.images.unsqueeze(0)
    return _as_fps(comp.frame_rate, default_fps), (
        _rgb8(to_uint8(images[i].detach().cpu().float().numpy())) for i in range(int(images.shape[0])))


def _file_source(value):
    """
    Source a decoder en flux, seulement pour une VIDEO qui EST un fichier sans
    decoupage (trim) : sinon get_stream_source() ignorerait le trim, ou
    reencoderait en H.264 (avec perte) une video faite d'images.
    """
    if type(value).__name__ != "VideoFromFile":
        return None
    try:
        if tuple(value.get_active_trim_window()) != (0.0, 0.0):
            return None
        return value.get_stream_source()
    except Exception:
        return None


def write_clip(frames, folder, prefix, workers=4):
    """
    Ecrit proxys JPEG et images pleine resolution PNG d'un clip, en flux.
    Renvoie (n, largeur, hauteur). ValueError si le clip est vide.
    """
    os.makedirs(folder, exist_ok=True)
    first = next(frames, None)
    if first is None:
        raise ValueError(f"{prefix.upper()} has no frames")
    h, w = first.shape[:2]
    pw, ph = proxy_size(w, h)

    def save(item):
        i, arr = item
        im = Image.fromarray(arr)
        im.save(os.path.join(folder, f"{prefix}_full_{i:06d}.png"), compress_level=PNG_LEVEL)
        if (pw, ph) != (w, h):
            im = im.resize((pw, ph), Image.LANCZOS)
        im.save(os.path.join(folder, f"{prefix}_{i:06d}.jpg"), quality=PROXY_QUALITY, subsampling=0)

    n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        pending = []
        for i, arr in enumerate(_chain(first, frames)):
            if arr.shape[:2] != (h, w):
                raise ValueError(f"{prefix.upper()}: frame {i} is {arr.shape[1]}x{arr.shape[0]}, "
                                 f"expected {w}x{h}")
            pending.append(ex.submit(save, (i, arr)))
            n += 1
            if len(pending) >= workers * 4:          # borne la memoire en vol
                pending.pop(0).result()
        for f in pending:
            f.result()
    return n, w, h


def _chain(first, rest):
    yield first
    yield from rest


def display_geometry(a, b):
    """Geometrie d'affichage commune : la plus grande des deux (en pixels)."""
    return (a["w"], a["h"]) if a["w"] * a["h"] >= b["w"] * b["h"] else (b["w"], b["h"])


def same_clock(fa, fb, rel=1e-3):
    """Deux fps a moins de 0,1 % l'un de l'autre (29.97 / 30000/1001) : une seule horloge."""
    return abs(fa - fb) <= rel * max(fa, fb)
