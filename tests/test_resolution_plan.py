"""
Panneau Resolution : le plan (nodes/resolution_plan.py) doit annoncer exactement
ce que fait la chaine (memes crops, memes tailles modele, meme ordre de regions).
"""

import random

import pytest

from chain import make_image, make_mask, random_config, run_chain, DEFAULTS
from nodes.resolution_plan import plan, report

SIZES = [(512, 512), (640, 480), (480, 640), (1000, 667), (777, 1333), (333, 251), (1536, 640)]


def _configs(n, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        c = random_config(rng)
        c["size"] = rng.choice(SIZES)
        out.append(c)
    return out


@pytest.mark.parametrize("c", _configs(40, 7), ids=[f"p{i}" for i in range(40)])
def test_plan_matches_chain(c):
    w, h = c["size"]
    mask = make_mask(w, h, c["mask"], seed=c["seed"], soft=c["soft"])
    kw = {k: v for k, v in c.items() if k in DEFAULTS}
    run = run_chain(make_image(w, h, seed=c["seed"]), mask, **kw)
    p = plan(w, h, mask, **kw)
    assert p["padded"] == [run["base"].shape[2], run["base"].shape[1]]
    assert len(p["regions"]) == len(run["infos"])
    for r, i in zip(p["regions"], run["infos"]):
        assert r["crop"] == [i["x"], i["y"], i["orig_width"], i["orig_height"]]
        assert r["model"] == [i["width"], i["height"]]
    assert report(p)
