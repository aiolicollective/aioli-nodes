"""
Nodes legacy (AioliMaskBBox, BBoxMultipleFix) : gardes a vie, ils doivent rendre
au bit pres comme en v1.7.1. Les empreintes ci-dessous ont ete calculees avec le
code v1.7.1 (commit cc0ce3c) ; float32, float16, float64, entrees 8 bits ou
continues, masque a la taille de l'image ou plus petit.
(Une autre version de Pillow pourrait changer le Lanczos : regenerer alors les
empreintes depuis la v1.7.1, ne pas les recopier depuis le code courant.)
"""

import contextlib
import hashlib
import io

import numpy as np
import pytest
import torch

from nodes import bbox_fix as BF
from nodes import mask_bbox as MB
from nodes.tensor_utils import to_uint8

GOLDEN_V171 = {
    'MB torch.float32-False-none-False-False': 'dcee9851e806cf55',
    'MB small torch.float32-False-none-False-False': 'bbf76a27e08f9cd4',
    'BF torch.float32-False-none-False-False': '959c8b44b47e2521',
    'MB torch.float32-False-1024-True-False': '4d32a8ed8cc5b878',
    'MB small torch.float32-False-1024-True-False': 'b760ca413e67432d',
    'BF torch.float32-False-1024-True-False': 'f8078591e202929d',
    'MB torch.float32-False-512-False-True': '291a1bc246648696',
    'MB small torch.float32-False-512-False-True': 'd806d5524d41e0bb',
    'BF torch.float32-False-512-False-True': 'fc3b323cc315c080',
    'MB torch.float32-True-none-False-False': '7183cc21082cd4f6',
    'MB small torch.float32-True-none-False-False': 'e0922e231e15b985',
    'BF torch.float32-True-none-False-False': '95a1ae0c56512428',
    'MB torch.float32-True-1024-True-False': '2b8a98cd8eaa71fa',
    'MB small torch.float32-True-1024-True-False': '1cab075b6ed2cdf6',
    'BF torch.float32-True-1024-True-False': '69802d03dc107d0f',
    'MB torch.float32-True-512-False-True': '601ad27033fbafae',
    'MB small torch.float32-True-512-False-True': 'fe617f5b656234ce',
    'BF torch.float32-True-512-False-True': '2fb627bcb05a7776',
    'MB torch.float16-False-none-False-False': 'd86e75a03553be7f',
    'MB small torch.float16-False-none-False-False': 'd4326cac730d06fd',
    'BF torch.float16-False-none-False-False': '22cbe214645701e9',
    'MB torch.float16-False-1024-True-False': '5237c839892c2463',
    'MB small torch.float16-False-1024-True-False': 'd0104288583b2131',
    'BF torch.float16-False-1024-True-False': 'b02e8206cc10b923',
    'MB torch.float16-False-512-False-True': '231f1579db7fb6b5',
    'MB small torch.float16-False-512-False-True': '0ccc7071abd02756',
    'BF torch.float16-False-512-False-True': '78a7a2936de30e0b',
    'MB torch.float16-True-none-False-False': '9794c7e58e81c618',
    'MB small torch.float16-True-none-False-False': 'c2b41e9a3f10fb4d',
    'BF torch.float16-True-none-False-False': '8f2a54bbbc53eddd',
    'MB torch.float16-True-1024-True-False': 'b9d0fb3cb23360e9',
    'MB small torch.float16-True-1024-True-False': '7ad4540004252a24',
    'BF torch.float16-True-1024-True-False': 'ed03fbb08ab21c93',
    'MB torch.float16-True-512-False-True': '19ddad8df17f4d3b',
    'MB small torch.float16-True-512-False-True': '7e912d8ccf65258e',
    'BF torch.float16-True-512-False-True': 'e658af30011de25b',
    'MB torch.float64-False-none-False-False': '218c10160733d8c0',
    'MB small torch.float64-False-none-False-False': '0e927f2a12b7411c',
    'BF torch.float64-False-none-False-False': 'f3e6581f3374d7ea',
    'MB torch.float64-False-1024-True-False': '89276bbf26a10fc5',
    'MB small torch.float64-False-1024-True-False': 'fa475642b000d752',
    'BF torch.float64-False-1024-True-False': '111657fb9fd25393',
    'MB torch.float64-False-512-False-True': '136d5df0f0bfbd92',
    'MB small torch.float64-False-512-False-True': '650be946a1d7bd55',
    'BF torch.float64-False-512-False-True': '892eadc9f88d53e6',
    'MB torch.float64-True-none-False-False': '3a61d3c8b67ec7ba',
    'MB small torch.float64-True-none-False-False': 'ff7d78d5c00fa79d',
    'BF torch.float64-True-none-False-False': 'dd02b90b61c4bc11',
    'MB torch.float64-True-1024-True-False': '8203db7134ecf46b',
    'MB small torch.float64-True-1024-True-False': 'ff03969fcce4103d',
    'BF torch.float64-True-1024-True-False': '60e803470ad7baa8',
    'MB torch.float64-True-512-False-True': 'a2d4b135cb2e1016',
    'MB small torch.float64-True-512-False-True': '5ddce2364e36b015',
    'BF torch.float64-True-512-False-True': '0146c86f9849be3c',
}


def digest(outs):
    h = hashlib.sha256()
    for o in outs:
        if torch.is_tensor(o):
            h.update(o.detach().cpu().contiguous().numpy().tobytes()); h.update(str(tuple(o.shape)).encode())
        else:
            h.update(repr(o).encode())
    return h.hexdigest()[:16]

def cases():
    g = torch.Generator().manual_seed(42)
    for dtype in (torch.float32, torch.float16, torch.float64):
        for cont in (False, True):
            img = torch.rand(1, 300, 420, 3, generator=g, dtype=torch.float32)
            if not cont:
                img = torch.round(img * 255) / 255
            img = img.to(dtype)
            mask = torch.zeros(1, 300, 420); mask[:, 60:200, 100:260] = 1 - torch.round(torch.rand(1, 140, 160, generator=g) * 255) / 255
            small = torch.nn.functional.interpolate(mask[None], size=(150, 210), mode="nearest")[0]
            for target, sq, ds in (("none", False, False), ("1024", True, False), ("512", False, True)):
                yield f"{dtype}-{cont}-{target}-{sq}-{ds}", img, mask, small, target, sq, ds



def _all():
    res = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for name, img, mask, small, target, sq, ds in cases():
            res["MB " + name] = digest(MB.AioliMaskBBox().execute(img, mask, 32, "16 (Flux)", target, sq, ds))
            res["MB small " + name] = digest(MB.AioliMaskBBox().execute(img, small, 32, "16 (Flux)", target, sq, ds))
            res["BF " + name] = digest(BF.BBoxMultipleFix().fix(img, mask, 90, 50, 180, 160, "16 (Flux)", target, sq, ds))
    return res


def test_legacy_bit_identical_to_v171():
    res = _all()
    assert set(res) == set(GOLDEN_V171)
    diff = {k: (res[k], GOLDEN_V171[k]) for k in res if res[k] != GOLDEN_V171[k]}
    assert not diff, diff


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_floor_is_old_expression(dtype):
    a = np.random.default_rng(0).random((64, 64, 3)).astype(dtype)
    assert np.array_equal(to_uint8(a, "floor"), (a * 255).clip(0, 255).astype(np.uint8))


def test_unknown_rounding_raises():
    with pytest.raises(ValueError):
        to_uint8(np.zeros(3), "ceil")
