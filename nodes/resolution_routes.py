"""
Route serveur du panneau Resolution : POST /aioli/resolution_plan

Corps JSON : {"image": nom du fichier du LoadImage (ex. "photo.png",
"pasted/x.png", "clipspace/....png [input]"), "use_mask": bool,
"settings": {scale, ratio, ...}}. Reponse : le plan (nodes/resolution_plan.py).

Lecture seule : le fichier est lu comme LoadImage le lit (orientation EXIF,
masque = 1 - alpha), dans les dossiers input / output / temp de ComfyUI
uniquement (folder_paths refuse les chemins qui en sortent).
"""

import asyncio
import os
from collections import OrderedDict

import numpy as np
import torch
from aiohttp import web
from PIL import Image, ImageOps

import folder_paths
from server import PromptServer

from .resolution_plan import DEFAULTS, plan_from_layout, region_layout, report

_CACHE = OrderedDict()      # (fichier, mtime, use_mask, scale, ratio, regen) -> layout
_CACHE_MAX = 16


def _load(name, use_mask):
    path = folder_paths.get_annotated_filepath(name)
    if not os.path.isfile(path):
        raise FileNotFoundError(name)
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        W, H = im.size
        mask = None
        if use_mask and "A" in im.getbands():
            a = np.array(im.getchannel("A")).astype(np.float32) / 255.0
            mask = (1.0 - torch.from_numpy(a))[None]
    return path, W, H, mask


def _compute(name, use_mask, settings):
    s = dict(DEFAULTS)
    s.update({k: settings[k] for k in DEFAULTS if k in settings})
    s["scale"] = float(s["scale"])
    for k in ("resolution", "padding"):
        s[k] = int(s[k])
    for k in ("regenerate_all", "force_square", "force_target_downscale"):
        s[k] = bool(s[k])
    path = folder_paths.get_annotated_filepath(name)
    key = (path, os.path.getmtime(path), bool(use_mask), s["scale"], s["ratio"], s["regenerate_all"])
    lay = _CACHE.get(key)
    if lay is None:
        _, W, H, mask = _load(name, use_mask)
        lay = region_layout(W, H, mask, s["scale"], s["ratio"], s["regenerate_all"])
        _CACHE[key] = lay
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
    else:
        _CACHE.move_to_end(key)
    p = plan_from_layout(lay, **s)
    p["settings"] = s
    p["text"] = report(p)
    p["origin"] = "preview"
    return p


@PromptServer.instance.routes.post("/aioli/resolution_plan")
async def resolution_plan(request):
    try:
        data = await request.json()
        name = str(data.get("image") or "")
        if not name:
            return web.json_response({"error": "pas d'image source"}, status=400)
        settings = data.get("settings") or {}
        loop = asyncio.get_running_loop()
        p = await loop.run_in_executor(None, _compute, name, bool(data.get("use_mask")), settings)
        return web.json_response(p)
    except (FileNotFoundError, ValueError) as e:
        return web.json_response({"error": f"image introuvable : {e}"}, status=404)
    except Exception as e:   # le panneau affiche l'erreur au lieu de planter
        return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=500)
