"""
Route serveur du panneau Resolution : POST /aioli/resolution_plan

Corps JSON : {"image": nom du fichier du LoadImage (ex. "photo.png",
"pasted/x.png", "clipspace/....png [input]"), "use_mask": bool,
"settings": {scale, ratio, ...}}. Reponse : le plan (nodes/resolution_plan.py).

- Reglages valides et bornes comme les widgets du panneau (400 sinon).
- Avec masque : l'image est lue par le LoadImage de ComfyUI lui-meme (meme
  decodage, meme masque = 1 - alpha) ; au-dela de MAX_PREVIEW_PIXELS (image x
  SCALE), pas de regions en apercu : elles sont donnees au run.
- Sans masque : seule la taille est lue (en-tete + orientation EXIF), sans
  decoder l'image (JPEG, MPO, WebP, PNG, GIF ; autres formats : LoadImage).
- Lecture seule, dans les dossiers input / output / temp de ComfyUI
  (folder_paths refuse les chemins qui en sortent).
"""

import asyncio
import os
import threading
from collections import OrderedDict

from aiohttp import web
from PIL import Image

import folder_paths
from server import PromptServer

from .ratio_outpaint import outpaint_layout
from .resolution_plan import (MAX_PREVIEW_PIXELS, plan_from_layout, region_layout, report,
                              scaled_size, validate_settings)

_CACHE = OrderedDict()      # (fichier, mtime, use_mask, scale, ratio, regen) -> layout
_CACHE_MAX = 16
_LOCK = threading.Lock()


class PreviewError(Exception):
    """Erreur affichee telle quelle par le panneau (statut 400 / 404)."""

    def __init__(self, msg, status=400):
        super().__init__(msg)
        self.status = status


_ROTATED = (5, 6, 7, 8)          # orientations EXIF a 90 degres : largeur et hauteur echangees


def _probe(path, name):
    """
    Taille comme LoadImage la voit, nombre d'images, et (W, H, masque) si
    LoadImage a deja du etre appele. Sans decoder pour JPEG / MPO / WebP / PNG /
    GIF (en-tete + orientation EXIF) ; autres formats (TIFF...) : LoadImage.
    """
    with Image.open(path) as im:
        W, H = im.size
        fmt = (im.format or "").upper()
        frames = getattr(im, "n_frames", 1)
        if fmt == "MPO":
            frames = 1                   # photo de telephone : LoadImage n'en garde qu'une
        orient = 1
        try:
            if fmt in ("JPEG", "MPO", "WEBP"):
                orient = im.getexif().get(0x0112, 1)
            elif fmt == "PNG" and "exif" in im.info:   # chunk eXIf avant l'image : lisible
                ex = Image.Exif()
                ex.load(im.info["exif"])
                orient = ex.get(0x0112, 1)
        except Exception:
            orient = 1
        if fmt in ("JPEG", "MPO", "WEBP", "PNG", "GIF"):
            if orient in _ROTATED:
                W, H = H, W
            return W, H, frames, None
    loaded = _load_with_mask(name)
    return loaded[0], loaded[1], frames, loaded


def _load_with_mask(name):
    """Image lue par le LoadImage de ComfyUI (meme decodage, masque = 1 - alpha)."""
    import nodes as comfy_nodes          # nodes.py de ComfyUI (pas le package du pack)
    image, mask = comfy_nodes.LoadImage().load_image(name)
    return int(image.shape[2]), int(image.shape[1]), mask


def _compute(name, use_mask, raw_settings):
    try:
        s = validate_settings(raw_settings)
    except ValueError as e:
        raise PreviewError(str(e), 400)
    try:
        path = folder_paths.get_annotated_filepath(name)
    except ValueError:
        raise PreviewError("image path refused", 400)
    if not os.path.isfile(path):
        raise PreviewError(f"image not found: {name}", 404)

    key = (path, os.path.getmtime(path), bool(use_mask), s["scale"], s["ratio"], s["regenerate_all"])
    with _LOCK:
        lay = _CACHE.get(key)
        if lay is not None:
            _CACHE.move_to_end(key)
    if lay is None:
        W, H, frames, loaded = _probe(path, name)
        mask = None
        if use_mask:
            # cout du chemin avec masque : canevas apres SCALE et outpaint, par image
            sw, sh = scaled_size(W, H, s["scale"])
            ol = outpaint_layout(sw, sh, s["ratio"], "none")
            PW, PH = (ol[0], ol[1]) if ol else (sw, sh)
            if frames > 1:
                raise PreviewError("animated image: mask regions are given at run time", 400)
            if PW * PH > MAX_PREVIEW_PIXELS:
                raise PreviewError(f"canvas {PW}x{PH} ({PW * PH / 1e6:.0f} MP): mask regions are "
                                   f"given at run time (preview limited to "
                                   f"{MAX_PREVIEW_PIXELS / 1e6:.0f} MP)", 400)
            W, H, mask = loaded or _load_with_mask(name)
            if not bool((mask > 0).any()):
                mask = None                   # pas d'alpha / masque vide : chemin direct
        lay = region_layout(W, H, mask, s["scale"], s["ratio"], s["regenerate_all"])
        with _LOCK:
            _CACHE[key] = lay
            while len(_CACHE) > _CACHE_MAX:
                _CACHE.popitem(last=False)
    p = plan_from_layout(lay, **s)
    p["settings"] = s
    p["text"] = report(p)
    p["origin"] = "preview"
    return p


@PromptServer.instance.routes.post("/aioli/resolution_plan")
async def resolution_plan(request):
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid request (JSON expected)"}, status=400)
    if not isinstance(data, dict):
        return web.json_response({"error": "invalid request"}, status=400)
    name = data.get("image")
    if not isinstance(name, str) or not name:
        return web.json_response({"error": "no source image"}, status=400)
    try:
        loop = asyncio.get_running_loop()
        p = await loop.run_in_executor(None, _compute, name, data.get("use_mask") is True,
                                       data.get("settings"))
        return web.json_response(p)
    except PreviewError as e:
        return web.json_response({"error": str(e)}, status=e.status)
    except Exception as e:   # le panneau affiche l'erreur au lieu de planter
        return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=500)
