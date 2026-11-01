"""
Test de la chaine complete des wf V2 (voir tests/chain.py), modele simule.

    pytest tests/                                # rapide (~2-3 min CPU)
    AIOLI_CHAIN_N=400 pytest tests/test_chain.py # plus de configurations tirees au sort

Promesse testee :
  - modele identite : 0 pixel modifie hors du masque combine, rien au-dela de
    mask_adjust + feather (+ frange du resize du masque) ;
  - aucune derive de couleur dans la zone recollee (< 1 niveau 8 bits en moyenne) ;
  - sans resize, recollage bit a bit.
Les defauts connus sont en xfail(strict) : quand ils seront corriges, le test
passera en XPASS et echouera, pour qu'on retire le xfail.
"""

import os
import random

import numpy as np
import pytest
import torch

from chain import (DEFAULTS, make_image, make_mask, make_model_shift, make_model_size,
                   measure, measure_shift, model_identity, model_vae_like,
                   random_config, run_chain, run_config)

N = int(os.environ.get("AIOLI_CHAIN_N", "40"))

SMALL_SIZES = [(512, 512), (640, 480), (480, 640), (1000, 667), (777, 1333), (333, 251), (1536, 640)]


def _small_configs(n, seed):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        c = random_config(rng)
        c["size"] = rng.choice(SMALL_SIZES)
        out.append(c)
    return out


CONFIGS = _small_configs(N, 20261101)


@pytest.mark.parametrize("cfg", CONFIGS, ids=[f"c{i}" for i in range(len(CONFIGS))])
def test_identity_promise(cfg):
    m = measure(run_config(cfg))
    assert m["outside_changed"] == 0, m
    assert m["beyond_reach"] == 0, m
    if m["inner_px"] > 0:
        assert abs(m["inner_bias"]) < 1.0, m           # derive de couleur moyenne
        assert m["inner_lp"] < 2.0, m                  # derive visible (basse frequence)


@pytest.mark.parametrize("mask_kind", ["blob", "edge", "multi", "line"])
@pytest.mark.parametrize("model", [model_identity, model_vae_like], ids=["identity", "vae_like"])
def test_color_fix_no_drift(mask_kind, model):
    src = make_image(1000, 667, seed=5)
    mask = make_mask(1000, 667, mask_kind, seed=5)
    m = measure(run_chain(src, mask, model=model, color_fix=True, resolution=1024))
    assert m["outside_changed"] == 0
    if m["inner_px"]:                      # un trait de 2 px n'a pas de zone pleine
        assert abs(m["inner_bias"]) < 1.0, m


@pytest.mark.parametrize("w, h, mult", [(640, 480, "16 (Flux)"), (512, 512, "64 (SDXL)"), (1024, 768, "8 (VAE minimum)")])
def test_no_resize_is_bit_exact(w, h, mult):
    """resolution 0, crop deja au multiple : ni resize ni quantification, recollage exact."""
    src = make_image(w, h, seed=2)
    mask = make_mask(w, h, "blob", seed=2)
    run = run_chain(src, mask, multiple=mult, resolution=0, force_target_downscale=False,
                    padding=0, feather=8, mask_adjust=6)
    assert all(i["orig_width"] == i["width"] and i["orig_height"] == i["height"] for i in run["infos"])
    d = (run["result"] - run["base"]).abs().max().item()
    assert d < 1e-6, d


def test_empty_mask_edits_whole_image():
    src = make_image(640, 480, seed=3)
    run = run_chain(src, make_mask(640, 480, "none"), resolution=1024)
    assert len(run["regions"]) == 1
    assert run["combined"].min().item() >= 0.999            # toute l'image recollee
    assert measure(run)["outside_changed"] == 0


def test_outpaint_keeps_source_pixels():
    """OUTPAINT RATIO seul : seules les bandes (+ feather) sont touchees."""
    src = make_image(1000, 667, seed=4)
    run = run_chain(src, make_mask(1000, 667, "none"), ratio="9:16", resolution=1024,
                    mask_adjust=0, feather=8)
    m = measure(run)
    assert m["outside_changed"] == 0 and m["beyond_reach"] == 0, m


def test_thin_region_stays_within_4_to_1():
    src = make_image(1536, 640, seed=6)
    run = run_chain(src, make_mask(1536, 640, "line", seed=6), resolution=1024, padding=0)
    for i in run["infos"]:
        assert max(i["width"], i["height"]) <= 4 * min(i["width"], i["height"])
    assert measure(run)["outside_changed"] == 0


@pytest.mark.parametrize("scale", [1.0, 0.5, 1.5])
def test_model_shift_is_magnified_by_crop_factor(scale):
    """
    Un modele qui glisse de 1 px (a sa taille) glisse de crop / modele px dans
    l'image finale : en image entiere, plus SCALE est grand, plus ca se voit.
    """
    src = make_image(2000, 1333, seed=7)
    run = run_chain(src, make_mask(2000, 1333, "none"), model=make_model_shift(1.0),
                    scale=scale, resolution=1024)
    s = measure_shift(run)
    assert abs(s["dx"] - s["factor"]) < 0.1 * s["factor"] + 0.1, s


# ---------------------------------------------------------------------------
# Defauts connus (audit v1.8) : xfail strict
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="liseres : le gris des bandes deborde dans l'image au resize "
                                       "(image entiere + SQUARE : ~7 niveaux sur ~16 px a x2.9)")
def test_no_band_bleed_full_image_square():
    src = make_image(3000, 2000, seed=3)
    m = measure(run_chain(src, make_mask(3000, 2000, "none"), resolution=1024, force_square=True))
    assert m["edge_lp"] < 4.0, m          # sans bande : ~1-3 niveaux (Lanczos seul)


@pytest.mark.xfail(strict=True, reason="batch > 1 : Color Fix et Assembler ne traitent que l'image 0")
def test_batch_is_preserved():
    src = make_image(640, 480, seed=1, batch=2)
    run = run_chain(src, make_mask(640, 480, "blob", seed=1), resolution=1024, color_fix=True)
    assert run["result"].shape[0] == 2


@pytest.mark.xfail(strict=True, raises=RuntimeError,
                   reason="base RGBA : l'assembleur plante sur le checker (palette 3 canaux)")
def test_rgba_base_does_not_crash():
    src = make_image(640, 480, seed=1, channels=4)
    run_chain(src, make_mask(640, 480, "blob", seed=1), resolution=1024)


def test_model_returning_other_aspect_is_stretched():
    """
    Un modele qui rend un autre ratio que le crop (ex. Gemini en aspect auto :
    1264x848 pour un crop 3:2 exact) est etire par l'assembleur. Mesure seulement :
    l'ecart reste a documenter (pas une regression du pack).
    """
    src = make_image(1500, 1000, seed=8)
    run = run_chain(src, make_mask(1500, 1000, "none"), model=make_model_size(1264, 848),
                    resolution=1024, force_square=False)
    i = run["infos"][0]
    stretch = (i["orig_width"] / 1264) / (i["orig_height"] / 848)
    assert abs(stretch - 1.0) < 0.01
