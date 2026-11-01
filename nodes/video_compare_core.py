"""
Coeur du 🎞️ Aioli Video Compare (fonctions pures, testables hors ComfyUI).

Lecture : sequence d'images sur <canvas>, synchro a l'image pres. Pour chaque
clip, chaque image est ecrite UNE fois dans le dossier temp de ComfyUI :
  - <side>_<i>.jpg      proxy (cote long <= PROXY_SIDE, JPEG q90 4:4:4) : lecture ;
  - <side>_full_<i>.png pleine resolution SANS PERTE (PNG, compression rapide) :
                        pause, pas a pas, zoom, difference.
Rien n'est garde en memoire apres l'execution. Une VIDEO est decodee par
ComfyUI lui-meme (get_components : trim, recadrage, rotation et conversion de
couleur identiques a la chaine) ; ce decodage tient le clip en float32 le
temps de l'ecriture, comme tout node video du coeur.

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
    uint8 (H, W, 3)). VIDEO : get_components() de ComfyUI (memes images que la
    chaine : trim, recadrage, rotation, conversion de couleur).
    """
    if isinstance(value, torch.Tensor):
        t = value if value.dim() == 4 else value.unsqueeze(0)
        return float(default_fps), (_rgb8(to_uint8(t[i].detach().cpu().float().numpy()))
                                    for i in range(int(t.shape[0])))
    comp = value.get_components()
    images = comp.images if comp.images.dim() == 4 else comp.images.unsqueeze(0)
    return _as_fps(comp.frame_rate, default_fps), (
        _rgb8(to_uint8(images[i].detach().cpu().float().numpy())) for i in range(int(images.shape[0])))


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


def same_clock(fa, fb, rel=1e-4):
    """
    Deux fps a moins de 0,01 % l'un de l'autre (29.97 saisi / 30000/1001) : une
    seule horloge. 24 et 23.976 sont deux cadences differentes (0,1 %).
    """
    return abs(fa - fb) <= rel * max(fa, fb)


def signature(value, fps):
    """
    Empreinte bon marche d'une entree, pour reutiliser les images deja ecrites
    quand le node est relance sans changement : forme + echantillon du contenu
    (IMAGE), ou source, trim, recadrage, taille et nombre d'images (VIDEO).
    """
    if isinstance(value, torch.Tensor):
        t = value.detach()
        step = max(1, int(t.shape[1]) // 64), max(1, int(t.shape[2]) // 64)
        sample = t[:, ::step[0], ::step[1]].float()
        return ("image", tuple(t.shape), str(t.dtype), round(float(sample.sum()), 4),
                round(float((sample * sample).sum()), 4), float(fps))
    parts = [type(value).__name__]
    for getter in ("get_stream_source", "get_active_trim_window", "get_dimensions",
                   "get_frame_count", "get_frame_rate"):
        try:
            v = getattr(value, getter)()
            parts.append(v if isinstance(v, (str, int, float, tuple)) else str(v))
        except Exception:
            parts.append(None)
    crop = getattr(value, "_VideoFromFile__crop", None)
    parts.append(str(crop))
    if isinstance(parts[1], str) and os.path.isfile(parts[1]):
        parts.append(os.path.getmtime(parts[1]))      # fichier reecrit sous le meme nom
    if parts[1] is None or not isinstance(parts[1], str):
        parts.append(id(value))          # pas de fichier : pas de reutilisation sure
    return tuple(parts)
