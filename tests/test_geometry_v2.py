"""
Invariants de la geometrie d'Aioli Mask BBox V2 (nodes/bbox_core_v2.py), math pur.

    pytest tests/test_geometry_v2.py                  # 20 000 cas tires au sort
    AIOLI_GEOM_N=300000 pytest tests/test_geometry_v2.py   # campagne longue (~4 min)
"""

import math
import os
import random

import pytest

from nodes.bbox_core_v2 import (CAP_SIDE, MAX_ASPECT, MULTIPLES, compute_crop_v2,
                                parse_multiple, pick_model_size)

N = int(os.environ.get("AIOLI_GEOM_N", "20000"))
RESOLUTIONS = [0, 512, 768, 1024, 1152, 1536, 2048, 2560, 4096]


def _random_case(rng):
    kind = rng.random()
    W, H = rng.randint(16, 8192), rng.randint(16, 8192)
    if kind < 0.15:                      # region tres fine
        if rng.random() < 0.5:
            bw, bh = rng.randint(1, 3), rng.randint(1, H)
        else:
            bw, bh = rng.randint(1, W), rng.randint(1, 3)
    elif kind < 0.25:                    # image entiere
        bw, bh = W, H
    else:
        bw, bh = rng.randint(1, W), rng.randint(1, H)
    bw, bh = min(bw, W), min(bh, H)
    if rng.random() < 0.3:               # collee a un bord / un coin
        x = rng.choice([0, W - bw])
        y = rng.choice([0, H - bh])
    else:
        x, y = rng.randint(0, W - bw), rng.randint(0, H - bh)
    return dict(x=x, y=y, width=bw, height=bh, src_w=W, src_h=H,
                multiple=rng.choice(MULTIPLES), resolution=rng.choice(RESOLUTIONS),
                force_square=rng.random() < 0.4, force_target_downscale=rng.random() < 0.5)


def check_invariants(a, c):
    mult = parse_multiple(a["multiple"])
    x, y, bw, bh, W, H = a["x"], a["y"], a["width"], a["height"], a["src_w"], a["src_h"]
    mw, mh = c["width"], c["height"]
    cx, cy, cw, ch = c["x"], c["y"], c["orig_width"], c["orig_height"]
    N = a["resolution"]

    # taille modele : multiples, jamais plus allongee que 4:1, carre si demande
    assert mw % mult == 0 and mh % mult == 0 and mw > 0 and mh > 0
    assert max(mw, mh) <= MAX_ASPECT * min(mw, mh) + 1e-9
    if a["force_square"]:
        assert mw == mh and cw == ch
    # plafond en pixels (tolerance 5 %)
    cap = max(CAP_SIDE * CAP_SIDE, N * N)
    native_ok = mw * mh <= cap * 1.05 or (cw, ch) == (mw, mh)
    assert native_ok, "taille modele au-dela du plafond"
    # le bbox n'est jamais coupe
    assert cx <= x and cy <= y and cx + cw >= x + bw and cy + ch >= y + bh
    # le crop reste dans l'image s'il y tient, sinon il la couvre (bandes)
    for o, ln, src in ((cx, cw, W), (cy, ch, H)):
        if ln <= src:
            assert 0 <= o <= src - ln
        else:
            assert o <= 0 and o + ln >= src
    # bandes et image_rect coherents
    l, t, r, b = c["bands"]
    assert (l, t, r, b) == (max(0, -cx), max(0, -cy), max(0, cx + cw - W), max(0, cy + ch - H))
    assert c["image_rect"] == (l, t, cw - r, ch - b)
    # jamais d'etirement au-dela de l'arrondi : ratio crop = ratio modele a 0,5 px pres
    if c["need_resize"]:
        s = c["scale"]
        assert abs(cw - mw * s) <= 0.5 + 1e-6 or cw == bw
        assert abs(ch - mh * s) <= 0.5 + 1e-6 or ch == bh
        sx, sy = mw / cw, mh / ch
        assert abs(sx / sy - 1.0) <= 1.0 / min(cw, ch) + 1e-9, "etirement anisotrope > 1 px"
        assert c["resize_label"] in ("upscale", "downscale")
    else:
        assert (cw, ch) == (mw, mh) and c["resize_label"] == ""


def test_geometry_random_cases():
    rng = random.Random(20261101)
    for _ in range(N):
        a = _random_case(rng)
        c = compute_crop_v2(**a)
        try:
            check_invariants(a, c)
        except AssertionError as e:
            pytest.fail(f"{e} : {a} -> {c}")


@pytest.mark.parametrize("ratio, area, mult", [
    (1.0, 1024 * 1024, 16), (1.5, 1024 * 1024, 16), (16 / 9, 2048 * 2048, 16),
    (0.25, 1024 * 1024, 64), (10.0, 1024 * 1024, 8), (1.0, 1152 * 1152, 64),
])
def test_pick_model_size(ratio, area, mult):
    w, h = pick_model_size(ratio, area, mult)
    assert w % mult == 0 and h % mult == 0
    assert max(w, h) <= 4 * min(w, h)
    if ratio == 1.0 and int(math.sqrt(area)) % mult == 0:
        assert w == h == int(math.sqrt(area))


def test_known_values():
    """Valeurs du README (equivalent carre a 2048, x16)."""
    for (bw, bh), expect in (((3000, 2000), (2496, 1664)), ((1600, 900), (2704, 1520))):
        c = compute_crop_v2(0, 0, bw, bh, bw, bh, "16 (Flux)", 2048, False, True)
        assert (c["width"], c["height"]) == expect


def test_empty_bbox_raises():
    with pytest.raises(ValueError):
        compute_crop_v2(0, 0, 0, 10, 100, 100, "16 (Flux)", 1024)
