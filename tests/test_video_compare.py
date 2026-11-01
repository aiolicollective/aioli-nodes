"""🎞️ Aioli Video Compare : fonctions pures (frames, proxys, pleine resolution)."""

import io
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from nodes.video_compare_core import describe, frames_of, full_png, proxy_size, write_proxies


def test_frames_of_image_and_video():
    img = torch.rand(5, 40, 60, 4)
    f, fps = frames_of(img, 24.0)
    assert f.shape == (5, 40, 60, 3) and fps == 24.0              # alpha retire, fps du widget
    video = SimpleNamespace(get_components=lambda: SimpleNamespace(images=torch.rand(3, 20, 30, 3),
                                                                    frame_rate=Fraction(30000, 1001)))
    f, fps = frames_of(video, 24.0)
    assert f.shape[0] == 3 and abs(fps - 29.97) < 0.01             # fps de la video
    d = describe(f, fps, "B")
    assert d["n"] == 3 and d["w"] == 30 and d["h"] == 20


def test_proxy_size():
    assert proxy_size(3840, 2160) == (1280, 720)
    assert proxy_size(640, 360) == (640, 360)                       # pas d'agrandissement
    assert proxy_size(1080, 1920) == (720, 1280)


def test_proxies_and_lossless_full(tmp_path):
    k = (np.arange(4 * 32 * 48 * 3) % 256).reshape(4, 32, 48, 3).astype(np.float32) / 255.0
    frames = torch.from_numpy(k)
    assert write_proxies(frames, str(tmp_path), "a", (48, 32)) == 4
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"a_{i:06d}.jpg" for i in range(4)]
    png = Image.open(io.BytesIO(full_png(frames, 2)))
    assert np.array_equal(np.array(png), np.round(k[2] * 255).astype(np.uint8))   # sans perte
    png_b = Image.open(io.BytesIO(full_png(frames, 1, (96, 64))))
    assert png_b.size == (96, 64)                                   # B dans la geometrie de A
