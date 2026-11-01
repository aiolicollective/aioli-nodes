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

import hashlib
import os
import weakref
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


def content_digest(images, workers=4):
    """
    Empreinte exacte d'un batch d'images (sha1 de chaque image, en parallele :
    ~5 Go/s sur 4 coeurs, 0,5 s pour 100 images 1080p en float32).
    """
    t = images.detach()
    frames = t if t.dim() == 4 else t.unsqueeze(0)

    def one(i):
        f = frames[i].cpu().contiguous()                          # une image a la fois (GPU -> RAM)
        raw = f.view(torch.uint8).numpy().reshape(-1)             # octets bruts, sans conversion
        return hashlib.sha1(raw).digest()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(one, range(int(frames.shape[0]))))
    head = repr((tuple(frames.shape), str(frames.dtype))).encode()
    return hashlib.sha1(head + b"".join(parts)).hexdigest()


def signature(value):
    """
    Identite d'une entree, pour reutiliser les images deja ecrites quand le
    node est relance sur le meme contenu :
      - IMAGE, ou video en memoire (VideoFromComponents) : empreinte exacte du
        contenu (une image modifiee sur place par un autre node est vue) ;
      - fichier video sur disque : chemin, taille, dates de modification et de
        changement (ns), inode, trim, recadrage. Linux / macOS : toute
        reecriture change la date de changement, meme `cp -p`. Windows : cette
        date est celle de creation ; un fichier reecrit sur place avec
        exactement la meme taille et la meme date de modification (ns) serait
        reutilise (cas d'outil de copie qui restaure les dates) ;
      - autre video en memoire : le MEME objet (reference faible, rien n'est
        retenu en memoire).
    Le fps d'une IMAGE n'en fait pas partie (donnee d'affichage).
    """
    if isinstance(value, torch.Tensor):
        return ("image", content_digest(value))
    name = type(value).__name__
    if name == "VideoFromComponents":
        try:
            comp = value.get_components()               # composants stockes, sans decodage
            return ("components", content_digest(comp.images), str(comp.frame_rate))
        except Exception:
            pass
    if name == "VideoFromFile":
        try:
            src = value.get_stream_source()          # chemin ou BytesIO, sans decodage
            if isinstance(src, str) and os.path.isfile(src):
                st = os.stat(src)
                return ("file", os.path.realpath(src), st.st_size, st.st_mtime_ns,
                        st.st_ctime_ns, st.st_ino, tuple(value.get_active_trim_window()),
                        str(getattr(value, "_VideoFromFile__crop", None)))
        except Exception:
            pass
    try:
        return ("object", weakref.ref(value))
    except TypeError:
        return None                                   # pas de reutilisation possible


def same_input(sig, value, current=None):
    """
    L'entree courante correspond-elle a cette identite ? `current` : la
    signature deja calculee de `value` (evite de la recalculer).
    """
    if sig is None:
        return False
    if sig[0] == "object":
        return sig[1]() is value
    return sig == (current if current is not None else signature(value))
