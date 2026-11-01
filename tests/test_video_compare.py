"""🎞️ Aioli Video Compare : fonctions pures (frames, ecriture des images, geometrie, horloge)."""

import os
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from nodes.video_compare_core import (display_geometry, iter_frames, proxy_size, same_clock,
                                      write_clip)


def test_iter_frames_image_and_video():
    fps, it = iter_frames(torch.rand(5, 40, 60, 4), 24.0)
    frames = list(it)
    assert fps == 24.0 and len(frames) == 5 and frames[0].shape == (40, 60, 3)   # alpha retire
    video = SimpleNamespace(get_components=lambda: SimpleNamespace(images=torch.rand(3, 20, 30, 3),
                                                                    frame_rate=Fraction(30000, 1001)))
    fps, it = iter_frames(video, 24.0)                  # pas un fichier : get_components()
    assert abs(fps - 29.97) < 0.01 and len(list(it)) == 3


def test_write_clip_lossless_and_proxies(tmp_path):
    k = (np.arange(4 * 32 * 48 * 3) % 256).reshape(4, 32, 48, 3).astype(np.float32) / 255.0
    _, it = iter_frames(torch.from_numpy(k), 24.0)
    assert write_clip(it, str(tmp_path), "a") == (4, 48, 32)
    names = sorted(os.listdir(tmp_path))
    assert names == [f"a_{i:06d}.jpg" for i in range(4)] + [f"a_full_{i:06d}.png" for i in range(4)]
    png = np.array(Image.open(tmp_path / "a_full_000002.png"))
    assert np.array_equal(png, np.round(k[2] * 255).astype(np.uint8))           # sans perte


def test_write_clip_empty_and_mixed_sizes(tmp_path):
    with pytest.raises(ValueError):
        write_clip(iter([]), str(tmp_path), "a")
    frames = iter([np.zeros((10, 10, 3), np.uint8), np.zeros((12, 10, 3), np.uint8)])
    with pytest.raises(ValueError):
        write_clip(frames, str(tmp_path), "b")


def test_geometry_and_clock():
    assert proxy_size(3840, 2160) == (1280, 720) and proxy_size(640, 360) == (640, 360)
    a, b = dict(w=1920, h=1080), dict(w=3840, h=2160)
    assert display_geometry(a, b) == (3840, 2160)                                # la plus grande
    assert same_clock(30000 / 1001, 29.97) and not same_clock(24.0, 48.0)
