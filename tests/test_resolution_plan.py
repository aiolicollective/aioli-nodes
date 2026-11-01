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


def test_validate_settings_bounds():
    from nodes.resolution_plan import validate_settings
    ok = validate_settings({"scale": 1.5, "ratio": "16:9", "resolution": 1536, "multiple": "64 (SDXL)",
                            "force_square": True, "force_target_downscale": False, "padding": 0,
                            "context": "full_image", "regenerate_all": True})
    assert ok["scale"] == 1.5 and ok["resolution"] == 1536
    for bad in ({"scale": 0}, {"scale": 9}, {"scale": "abc"}, {"resolution": 10 ** 9},
                {"padding": -1}, {"ratio": "2:1"}, {"multiple": "12"}, {"context": "x"},
                {"force_square": "false"}, {"scale": True}, {"resolution": float("inf")},
                {"scale": float("nan")}):
        with pytest.raises(ValueError):
            validate_settings(bad)
    for bad in ("x", [1], 3):
        with pytest.raises(ValueError):
            validate_settings(bad)


def test_quiet_is_per_thread():
    """L'apercu fait taire les nodes sans toucher sys.stdout ni les autres threads."""
    import io
    import sys
    import threading
    from contextlib import redirect_stdout
    from nodes.log import log, quiet
    import torch
    stdout = sys.stdout
    m = torch.zeros(1, 300, 400)
    m[:, 50:120, 60:200] = 1
    threads = [threading.Thread(target=lambda: plan(400, 300, m, scale=1.5, ratio="16:9")) for _ in range(8)]
    for t in threads:
        t.start()
    buf = io.StringIO()
    with redirect_stdout(buf):
        log("visible")
        with quiet():
            log("cache")
    for t in threads:
        t.join()
    assert sys.stdout is stdout
    assert buf.getvalue() == "visible\n"
