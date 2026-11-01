"""
BBox Multiple Assembler : le calcul du masque sur la fenetre du crop (v1.8)
donne le meme resultat, au bit pres, que le calcul sur toute la toile (v1.7).
"""

import random
import time

import pytest
import torch

from nodes.bbox_assembler import BBoxMultipleAssembler


def _case(rng, batch=1):
    W, H = rng.randint(64, 700), rng.randint(64, 700)
    base = torch.rand(batch, H, W, 3)
    crops, masks, xs, ys, ws, hs = [], [], [], [], [], []
    for _ in range(rng.randint(1, 4)):
        w, h = rng.randint(8, W + 120), rng.randint(8, H + 120)
        x, y = rng.randint(-60, W - 4), rng.randint(-60, H - 4)    # bandes possibles
        mw, mh = (w, h) if rng.random() < 0.4 else (rng.randint(8, 300), rng.randint(8, 300))
        crops.append(torch.rand(rng.choice([1, batch]), mh, mw, 3))
        kind = rng.random()
        if kind < 0.3:
            m = torch.ones(1, mh, mw)
        elif kind < 0.6:                       # masque doux (pinceau, LoadImage)
            m = torch.nn.functional.avg_pool2d((torch.rand(1, 1, mh, mw) > 0.6).float(), 5, 1, 2)[0]
            m = 1.0 - torch.round((1.0 - m) * 255) / 255
        else:
            m = (torch.rand(1, mh, mw) > 0.6).float()
        masks.append(m)
        xs.append(x); ys.append(y); ws.append(w); hs.append(h)
    kw = dict(order=[rng.choice(["list_first_on_top", "area_large_under"])],
              mask_adjust=[rng.choice([0, 1, 6, 20, 24, -3, -24])],
              feather=[rng.choice([0, 1, 5, 8, 24])], opacity=[rng.choice([1.0, 0.7])])
    return [base], crops, masks, xs, ys, ws, hs, kw


def _run(args, window):
    A = BBoxMultipleAssembler
    old = A._use_window
    A._use_window = window
    try:
        base, crops, masks, xs, ys, ws, hs, kw = args
        return A().assemble(base, crops, masks, xs, ys, ws, hs, **kw)
    finally:
        A._use_window = old


@pytest.mark.parametrize("seed", range(60))
def test_window_is_bit_identical(seed, capsys):
    rng = random.Random(seed)
    args = _case(rng, batch=rng.choice([1, 1, 2]))
    a = _run(args, True)
    b = _run(args, False)
    for ta, tb, name in zip(a, b, ("image", "combined_mask", "checker")):
        assert torch.equal(ta, tb), name


def test_window_is_faster():
    rng = random.Random(0)
    W, H = 3000, 2000
    base = [torch.rand(1, H, W, 3)]
    crops = [torch.rand(1, 512, 512, 3) for _ in range(4)]
    masks = [torch.ones(1, 512, 512) for _ in range(4)]
    xs, ys = [100, 900, 1700, 2300], [100, 600, 1100, 1400]
    ws = hs = [500] * 4
    kw = dict(order=["list_first_on_top"], mask_adjust=[6], feather=[8], opacity=[1.0])
    t = time.time(); _run((base, crops, masks, xs, ys, ws, hs, kw), True); tw = time.time() - t
    t = time.time(); _run((base, crops, masks, xs, ys, ws, hs, kw), False); tf = time.time() - t
    print(f"fenetre {tw:.2f}s, toile entiere {tf:.2f}s")
    assert tw < tf


def test_rgba_base_with_rgb_crop_keeps_alpha():
    """Crop RVB (VAE Decode) sur base RGBA : l'alpha de la base est garde."""
    base = torch.rand(1, 120, 160, 4)
    base[..., 3] = 0.8
    crop = torch.rand(1, 64, 64, 3)
    out, comb, checker = BBoxMultipleAssembler().assemble(
        [base], [crop], [torch.ones(1, 64, 64)], [20], [20], [64], [64],
        order=["list_first_on_top"], mask_adjust=[0], feather=[0], opacity=[1.0])
    assert out.shape[-1] == 4 and checker.shape[-1] == 4
    assert torch.allclose(out[..., 3], base[..., 3])


def test_batch_equals_separate_runs():
    """Batch : l'image b de sortie = un appel sur la base b et le crop b."""
    rng = random.Random(3)
    base, crops, masks, xs, ys, ws, hs, kw = _case(rng, batch=3)
    crops = [torch.rand(3, c.shape[1], c.shape[2], 3) for c in crops]
    full = BBoxMultipleAssembler().assemble(base, crops, masks, xs, ys, ws, hs, **kw)
    for b in range(3):
        one = BBoxMultipleAssembler().assemble([base[0][b:b + 1]], [c[b:b + 1] for c in crops],
                                               masks, xs, ys, ws, hs, **kw)
        for t_full, t_one in zip(full, one):
            assert torch.equal(t_full[b], t_one[0])


@pytest.mark.parametrize("adj, fea", [(0, 8), (6, 8), (0, 5), (-8, 8), (-5, 5), (-7, 7), (-9, 9),
                                      (20, 30), (13, 24), (6, 40), (0, 64), (0, 100), (-40, 16), (0, 0), (0, 1)])
def test_feather_goes_outside(adj, fea):
    """
    Fondu vers l'exterieur : alpha = 1 dans tout le masque (apres mask_adjust),
    fondu dehors, 0 au-dela de mask_adjust + feather + 1 (distance Chebyshev).
    mask_adjust = -feather : rien hors du trait dessine.
    """
    from scipy import ndimage
    H, W = 200, 260
    m = torch.zeros(1, 120, 140)
    m[:, 30:90, 40:100] = 1.0
    base = torch.rand(1, H, W, 3)
    _, comb, _ = BBoxMultipleAssembler().assemble(
        [base], [torch.rand(1, 120, 140, 3)], [m], [50], [40], [140], [120],
        order=["list_first_on_top"], mask_adjust=[adj], feather=[fea], opacity=[1.0])
    a = comb[0].numpy()
    drawn = (torch.zeros(H, W).index_put_((torch.arange(70, 130).repeat_interleave(60),
                                            torch.arange(90, 150).repeat(60)), torch.tensor(1.0)) > 0).numpy()
    inside = ndimage.distance_transform_cdt(drawn, metric="chessboard")       # profondeur dans le trait
    outside = ndimage.distance_transform_cdt(~drawn, metric="chessboard")     # distance hors du trait
    if adj >= 0:
        assert a[drawn].min() == 1.0                                          # 100 % dans le trait
    assert a[outside > max(0, adj) + fea].max() == 0.0
    if adj == -fea:
        assert a[~drawn].max() == 0.0                                         # rien hors du trait


@pytest.mark.parametrize("adj, fea", [(0, 8), (13, 24), (6, 40), (0, 64), (0, 100), (20, 30), (-40, 16),
                                      (-5, 5), (-9, 9), (3, 7)])
def test_feather_rule_direct(adj, fea):
    """
    Regle du fondu sur alpha_from_mask directement (sans la coupe du crop),
    chemins plein resolution et sous-echantillonne : 1 dans le masque apres
    mask_adjust, 0 au-dela de mask_adjust + feather. Masque en L qui touche
    le bord de l'image.
    """
    from scipy import ndimage
    from nodes.bbox_assembler import _grow_shrink_fast, alpha_from_mask
    H, W = 400, 460
    m = torch.zeros(H, W)
    m[150:300, 0:120] = 1.0
    m[150:200, 0:260] = 1.0
    a = alpha_from_mask(m, adj, fea).numpy()
    inside = _grow_shrink_fast(m, adj).numpy() >= 1.0
    drawn = m.numpy() > 0
    out = ndimage.distance_transform_cdt(~drawn, metric="chessboard")
    if inside.any():
        assert a[inside].min() == 1.0
    assert a[out > max(0, adj) + fea].max() == 0.0
