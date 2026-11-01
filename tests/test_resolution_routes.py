"""
Route /aioli/resolution_plan hors ComfyUI : server, folder_paths et le
LoadImage de ComfyUI sont remplaces par des doublures minimales. Attrape les
erreurs de la route (nom manquant, statut faux) sans lancer de serveur.
"""

import importlib
import sys
import types

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture()
def routes(tmp_path, monkeypatch):
    class _Routes:
        def post(self, path):
            return lambda fn: fn
    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=_Routes()))
    fp = types.ModuleType("folder_paths")

    def get_annotated_filepath(name):
        if ".." in name or name.startswith("/"):
            raise ValueError("Invalid file path")
        return str(tmp_path / name)
    fp.get_annotated_filepath = get_annotated_filepath

    class LoadImage:                      # doublure : meme convention que ComfyUI
        def load_image(self, name):
            im = Image.open(tmp_path / name)
            rgb = torch.from_numpy(np.array(im.convert("RGB")).astype(np.float32) / 255)[None]
            if "A" in im.getbands():
                a = np.array(im.getchannel("A")).astype(np.float32) / 255
                return rgb, (1.0 - torch.from_numpy(a))[None]
            return rgb, torch.zeros((1, 64, 64))
    comfy_nodes = types.ModuleType("nodes")
    comfy_nodes.LoadImage = LoadImage
    monkeypatch.setitem(sys.modules, "server", server)
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    rgba = np.zeros((300, 400, 4), np.uint8)
    rgba[..., 3] = 255
    rgba[60:140, 80:200, 3] = 0           # zone dessinee (masque = 1 - alpha)
    Image.fromarray(rgba, "RGBA").save(tmp_path / "m.png")
    Image.fromarray(rgba[..., :3]).save(tmp_path / "rgb.png")
    import nodes.resolution_routes as R
    R = importlib.reload(R)
    R._CACHE.clear()
    monkeypatch.setattr(R, "_load_with_mask", lambda name: _load(LoadImage, name))
    return R


def _load(cls, name):
    image, mask = cls().load_image(name)
    return int(image.shape[2]), int(image.shape[1]), mask


def test_route_with_mask(routes):
    p = routes._compute("m.png", True, {"scale": 1.5, "ratio": "16:9", "resolution": 1024,
                                        "force_square": False})
    kinds = [r["kind"] for r in p["regions"]]
    assert kinds == ["masque", "outpaint"]
    assert p["scaled"] == [600, 450] and p["text"]


def test_route_without_mask_and_errors(routes):
    p = routes._compute("rgb.png", True, {})          # pas d'alpha : chemin direct
    assert [r["kind"] for r in p["regions"]] == ["image entière"]
    for name, settings, status in (("m.png", {"scale": 0}, 400), ("../x.png", {}, 400),
                                   ("absent.png", {}, 404), ("m.png", "x", 400)):
        with pytest.raises(routes.PreviewError) as e:
            routes._compute(name, False, settings)
        assert e.value.status == status
