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
  decoder l'image.
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


def _size_only(path):
    """Taille comme LoadImage la voit (orientation EXIF), sans decoder l'image."""
    with Image.open(path) as im:
        W, H = im.size
        try:
            if im.getexif().get(0x0112, 1) in (5, 6, 7, 8):   # rotation de 90 degres
                W, H = H, W
        except Exception:
            pass
    return W, H


def _load_with_mask(name):
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
        raise PreviewError("chemin d'image refuse", 400)
    if not os.path.isfile(path):
        raise PreviewError(f"image introuvable : {name}", 404)

    key = (path, os.path.getmtime(path), bool(use_mask), s["scale"], s["ratio"], s["regenerate_all"])
    with _LOCK:
        lay = _CACHE.get(key)
        if lay is not None:
            _CACHE.move_to_end(key)
    if lay is None:
        W, H = _size_only(path)
        mask = None
        if use_mask:
            sw, sh = scaled_size(W, H, s["scale"])
            if sw * sh > MAX_PREVIEW_PIXELS:
                raise PreviewError(f"image x SCALE = {sw}x{sh} ({sw * sh / 1e6:.0f} MP) : "
                                   f"regions du masque donnees au run (apercu limite a "
                                   f"{MAX_PREVIEW_PIXELS / 1e6:.0f} MP)", 400)
            W, H, mask = _load_with_mask(name)
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
        return web.json_response({"error": "requete invalide (JSON attendu)"}, status=400)
    if not isinstance(data, dict):
        return web.json_response({"error": "requete invalide"}, status=400)
    name = data.get("image")
    if not isinstance(name, str) or not name:
        return web.json_response({"error": "pas d'image source"}, status=400)
    try:
        loop = asyncio.get_running_loop()
        p = await loop.run_in_executor(None, _compute, name, data.get("use_mask") is True,
                                       data.get("settings"))
        return web.json_response(p)
    except PreviewError as e:
        return web.json_response({"error": str(e)}, status=e.status)
    except Exception as e:   # le panneau affiche l'erreur au lieu de planter
        return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=500)
