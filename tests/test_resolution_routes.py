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
    assert kinds == ["mask", "outpaint"]
    assert p["scaled"] == [600, 450] and p["text"]


def test_route_without_mask_and_errors(routes):
    p = routes._compute("rgb.png", True, {})          # pas d'alpha : chemin direct
    assert [r["kind"] for r in p["regions"]] == ["whole image"]
    for name, settings, status in (("m.png", {"scale": 0}, 400), ("../x.png", {}, 400),
                                   ("absent.png", {}, 404), ("m.png", "x", 400)):
        with pytest.raises(routes.PreviewError) as e:
            routes._compute(name, False, settings)
        assert e.value.status == status


def test_route_probe_formats(routes, tmp_path):
    """Taille comme LoadImage : orientation EXIF (JPEG, PNG eXIf), MPO = 1 image, GIF sans decoder."""
    rgb = np.zeros((200, 300, 3), np.uint8)
    ex = Image.Exif()
    ex[0x0112] = 6
    Image.fromarray(rgb).save(tmp_path / "o6.jpg", exif=ex.tobytes())
    Image.fromarray(rgb).save(tmp_path / "o6.png", exif=ex.tobytes())
    Image.fromarray(rgb).save(tmp_path / "a.gif", save_all=True,
                              append_images=[Image.fromarray(rgb + 90)])   # frames distinctes
    for name, size, frames in (("o6.jpg", (200, 300), 1), ("o6.png", (200, 300), 1),
                               ("a.gif", (300, 200), 2)):
        W, H, n, loaded = routes._probe(str(tmp_path / name), name)
        assert (W, H, n) == (*size, frames) and loaded is None, name


def test_route_caps(routes, tmp_path):
    """Plafond sur le canevas apres outpaint ; image animee refusee avec masque."""
    big = np.zeros((10, 5000, 4), np.uint8)               # 5000x10 : petit, mais...
    Image.fromarray(big, "RGBA").save(tmp_path / "wide.png")
    with pytest.raises(routes.PreviewError) as e:          # ... 9:16 -> 5000x8896 = 44 MP
        routes._compute("wide.png", True, {"ratio": "9:16"})
    assert e.value.status == 400 and "canvas" in str(e.value)
    rgb = np.zeros((50, 60, 3), np.uint8)
    Image.fromarray(rgb).save(tmp_path / "anim.gif", save_all=True, append_images=[Image.fromarray(rgb + 90)])
    with pytest.raises(routes.PreviewError):
        routes._compute("anim.gif", True, {})
    assert routes._compute("anim.gif", False, {})["regions"]   # sans masque : apercu possible
