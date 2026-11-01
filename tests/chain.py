"""
Simulation de la chaine complete des wf V2, hors ComfyUI.

    LoadImage -> ImageScaleBy (SCALE, lanczos) -> Ratio Outpaint Calc (multiple = none)
      -> Mask Split Regions (mask_padded) -> Region Mask List (extra_mask = bandes
         ou image entiere, empty = full_image)
      -> Aioli Mask BBox V2 (une passe par region)
      -> modele SIMULE (identite, decalage, zoom...)
      -> Inpaint Color Fix (measure_area = image_area), optionnel
      -> BBox Multiple Assembler

Meme cablage que aioli_INPAINT_Img_*_v2.json. Les nodes sont appeles comme
ComfyUI le fait (listes pour INPUT_IS_LIST, map par region pour les autres).

Usage :
    python tests/chain.py --n 400            # rapport sur 400 configs tirees au sort
    python tests/chain.py --shift            # mesure du raccord avec un modele qui glisse / zoome
Les tests pytest (tests/test_chain.py) reutilisent ces fonctions.
"""

import argparse
import contextlib
import io
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from nodes.ratio_outpaint import RatioOutpaintCalc          # noqa: E402
from nodes.mask_split_regions import MaskSplitRegions       # noqa: E402
from nodes.region_mask_list import RegionMaskList           # noqa: E402
from nodes.mask_bbox_v2 import AioliMaskBBoxV2              # noqa: E402
from nodes.inpaint_color_fix import InpaintColorFix         # noqa: E402
from nodes.bbox_assembler import BBoxMultipleAssembler      # noqa: E402


MULTIPLES = ["8 (VAE minimum)", "16 (Flux)", "32 (SD1.5)", "64 (SDXL)"]
RATIOS = ["none", "1:1", "4:5", "5:4", "3:4", "4:3", "16:9", "9:16"]
LEVEL = 255.0


# ---------------------------------------------------------------------------
# Entrees synthetiques
# ---------------------------------------------------------------------------

def make_image(w, h, seed=0, channels=3, batch=1):
    """
    Image 8 bits (valeurs k/255 exactes, comme LoadImage) : degrades, aplats,
    traits fins, texture 1/f. Assez de contenu pour qu'un decalage se voie.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = []
    for b in range(batch):
        img = np.zeros((h, w, 3), np.float32)
        img[..., 0] = 0.15 + 0.7 * xx / max(1, w - 1)
        img[..., 1] = 0.15 + 0.7 * yy / max(1, h - 1)
        img[..., 2] = 0.5 + 0.25 * np.sin(xx / 37.0 + b) * np.cos(yy / 53.0)
        # texture 1/f (bruit basse frequence + detail)
        for octave, amp in ((64, 0.10), (16, 0.06), (4, 0.04), (1, 0.03)):
            gh, gw = max(2, h // octave + 2), max(2, w // octave + 2)
            n = rng.standard_normal((gh, gw, 3)).astype(np.float32)
            n = np.array(Image.fromarray(((n * 0.25 + 0.5).clip(0, 1) * 255).astype(np.uint8))
                         .resize((w, h), Image.BICUBIC)).astype(np.float32) / 255.0 - 0.5
            img += amp * n
        # traits fins (1-2 px) : revelent tout glissement. Pas sur la 1re ligne /
        # colonne : un trait colle au bord est ecrete par tout Lanczos aller-retour
        # (jusqu'a ~10 niveaux), ce qui masquerait les defauts propres au pack.
        # Contraste modere : un trait noir pur a 0 serait ecrete par le ringing
        # du Lanczos (hors pack) et fausserait la mesure de derive.
        for k in range(13, w - 13, 97):
            img[:, k:k + 1, :] = img[:, k:k + 1, :] * 0.55 + 0.05
        for k in range(11, h - 13, 89):
            img[k:k + 2, :, :] = img[k:k + 2, :, :] * 0.55 + 0.40
        img = np.clip(img, 0, 1)
        if channels == 4:
            img = np.concatenate([img, np.ones((h, w, 1), np.float32)], axis=-1)
        img = np.round(img * 255) / 255.0
        out.append(img.astype(np.float32))
    return torch.from_numpy(np.stack(out))


def make_mask(w, h, kind, seed=0, soft=False):
    """Masque dessine (B=1). kind : none, blob, edge, corner, line, multi, full, tiny."""
    rng = np.random.default_rng(seed)
    m = np.zeros((h, w), np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    def ellipse(cx, cy, rx, ry):
        return ((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2 <= 1.0

    if kind == "blob":
        cx, cy = rng.uniform(0.25, 0.75) * w, rng.uniform(0.25, 0.75) * h
        m[ellipse(cx, cy, rng.uniform(0.05, 0.25) * w, rng.uniform(0.05, 0.25) * h)] = 1
    elif kind == "edge":          # touche un bord
        side = rng.integers(4)
        cx = [0, w - 1, rng.uniform(0.3, 0.7) * w, rng.uniform(0.3, 0.7) * w][side]
        cy = [rng.uniform(0.3, 0.7) * h, rng.uniform(0.3, 0.7) * h, 0, h - 1][side]
        m[ellipse(cx, cy, 0.15 * w, 0.15 * h)] = 1
    elif kind == "corner":
        cx, cy = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)][rng.integers(4)]
        m[ellipse(cx, cy, 0.2 * w, 0.2 * h)] = 1
    elif kind == "line":          # region tres fine (2 px), tres allongee
        y0 = int(rng.uniform(0.2, 0.8) * h)
        x0, x1 = int(0.1 * w), int(0.9 * w)
        m[y0:y0 + 2, x0:x1] = 1
    elif kind == "multi":
        for _ in range(3):
            cx, cy = rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h
            m[ellipse(cx, cy, rng.uniform(0.03, 0.1) * w, rng.uniform(0.03, 0.1) * h)] = 1
    elif kind == "tiny":          # quelques pixels (au-dessus de min_area = 64)
        cx, cy = int(rng.uniform(0.2, 0.8) * w), int(rng.uniform(0.2, 0.8) * h)
        m[cy:cy + 9, cx:cx + 9] = 1
    elif kind == "full":
        m[:] = 1
    if soft and m.any():
        m = ndimage.gaussian_filter(m, 2.0).astype(np.float32)
        m = np.round(m * 255) / 255.0
    return torch.from_numpy(m)[None]


# ---------------------------------------------------------------------------
# Equivalents des nodes ComfyUI hors pack
# ---------------------------------------------------------------------------

def image_scale_by(img, scale):
    """ImageScaleBy (lanczos) : taille arrondie, PIL Lanczos sur uint8."""
    if scale == 1.0:
        return img
    B, H, W, C = img.shape
    nw, nh = round(W * scale), round(H * scale)
    out = []
    for b in range(B):
        arr = np.clip(255.0 * img[b].numpy(), 0, 255).astype(np.uint8)
        out.append(np.array(Image.fromarray(arr).resize((nw, nh), Image.LANCZOS)).astype(np.float32) / 255.0)
    return torch.from_numpy(np.stack(out))


def solid_mask(h, w, value=1.0):
    return torch.full((1, h, w), float(value))


# ---------------------------------------------------------------------------
# Modeles simules
# ---------------------------------------------------------------------------

def _warp(img, dx=0.0, dy=0.0, zoom=1.0):
    """Decalage (dx, dy en px modele) et zoom autour du centre, bilineaire, bord replique."""
    B, H, W, C = img.shape
    t = img.permute(0, 3, 1, 2)
    theta = torch.tensor([[1.0 / zoom, 0.0, -2.0 * dx / W],
                          [0.0, 1.0 / zoom, -2.0 * dy / H]], dtype=torch.float32)
    grid = F.affine_grid(theta.unsqueeze(0).expand(B, 2, 3), (B, C, H, W), align_corners=False)
    out = F.grid_sample(t, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out.permute(0, 2, 3, 1).contiguous()


def model_identity(crop, mask, info):
    return crop


def model_vae_like(crop, mask, info):
    """Identite + bruit continu (sortie VAE : plus de valeurs k/255 exactes)."""
    g = torch.Generator().manual_seed(1234)
    return (crop + torch.randn(crop.shape, generator=g) * (0.5 / LEVEL)).clamp(0, 1)


def make_model_shift(dx, dy=0.0):
    def model(crop, mask, info):
        return _warp(crop, dx, dy, 1.0)
    model.__name__ = f"shift({dx},{dy})"
    return model


def make_model_zoom(z):
    def model(crop, mask, info):
        return _warp(crop, 0.0, 0.0, 1.0 + z)
    model.__name__ = f"zoom({z})"
    return model


def make_model_size(nw, nh):
    """Modele qui ne rend pas la taille demandee (ex. Gemini en aspect auto)."""
    def model(crop, mask, info):
        arr = (crop[0].numpy() * 255).round().astype(np.uint8)
        out = np.array(Image.fromarray(arr).resize((nw, nh), Image.LANCZOS)).astype(np.float32) / 255.0
        return torch.from_numpy(out)[None]
    model.__name__ = f"size({nw}x{nh})"
    return model


# ---------------------------------------------------------------------------
# La chaine
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    scale=1.0, ratio="none", regenerate_all=False,
    padding=48, multiple="16 (Flux)", resolution=1024,
    force_square=False, force_target_downscale=True, context="region",
    color_fix=False, cf_threshold=-1.0, cf_blend=1.0, cf_feather=7,
    order="list_first_on_top", mask_adjust=6, feather=8,
)


def run_chain(src, drawn_mask, model=model_identity, quiet=True, **kw):
    cfg = dict(DEFAULTS)
    cfg.update(kw)
    sink = io.StringIO()
    ctx = contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()
    with ctx:
        img = image_scale_by(src, cfg["scale"])
        image_padded, bands, mask_padded = RatioOutpaintCalc().calculate(
            img, cfg["ratio"], "none", mask=drawn_mask)
        split, _count = MaskSplitRegions().split(mask_padded, empty_mask="nothing")
        H, W = image_padded.shape[1], image_padded.shape[2]
        extra = solid_mask(H, W) if cfg["regenerate_all"] else bands
        regions, _bg, _n = RegionMaskList().build(
            list(split), add_background=[False], threshold=[0.5], min_bg_area=[0.0],
            extra_mask=[extra], empty=["full_image"])

        bbox = AioliMaskBBoxV2()
        cf = InpaintColorFix()
        crops, masks, xs, ys, ws, hs, infos = [], [], [], [], [], [], []
        for reg in regions:
            (ic, mc, x, y, ow, oh, mw, mh, tsz, area) = bbox.execute(
                image_padded, reg, cfg["padding"], cfg["multiple"], cfg["resolution"],
                cfg["force_square"], cfg["force_target_downscale"], cfg["context"])
            info = dict(x=x, y=y, orig_width=ow, orig_height=oh, width=mw, height=mh)
            out = model(ic, mc, info)
            if cfg["color_fix"]:
                out, _cm = cf.fix(ic, out, cfg["cf_threshold"], cfg["cf_blend"], cfg["cf_feather"],
                                  measure_area=area)
            crops.append(out)
            masks.append(mc)
            xs.append(x); ys.append(y); ws.append(ow); hs.append(oh)
            infos.append(info)

        result, combined, _checker = BBoxMultipleAssembler().assemble(
            [image_padded], crops, masks, xs, ys, ws, hs,
            order=[cfg["order"]], mask_adjust=[cfg["mask_adjust"]],
            feather=[cfg["feather"]], opacity=[1.0])

    return dict(cfg=cfg, base=image_padded, bands=bands, regions=regions, infos=infos,
                result=result, combined=combined, log=sink.getvalue())


# ---------------------------------------------------------------------------
# Mesures
# ---------------------------------------------------------------------------

def _lowpass(a, sigma):
    """Passe-bas rapide (boite ~ gaussienne de meme ecart-type), cout independant de sigma."""
    k = max(1, int(round(sigma * 3.46)) | 1)
    size = (k, k, 1) if a.ndim == 3 else (k, k)
    return ndimage.uniform_filter(a, size=size, mode="nearest")


def measure(run):
    """
    Mesures sur l'image de travail (source x SCALE, + bandes d'outpaint) :

    outside_changed  pixels modifies la ou combined_mask == 0 (doit etre 0)
    spill_px         plus grande distance (px, Chebyshev : grow et feather sont
                     des noyaux carres) entre un pixel modifie et la zone
                     dessinee (regions >= 0.5)
    reach_px         borne attendue : mask_adjust + feather + frange du resize du
                     masque (deux Lanczos de rayon 3 : 3 px + 3 px x crop / modele)
    beyond_reach     pixels modifies au-dela de reach_px
    inner_*          ecart dans la zone pleinement recollee (combined >= 0.999),
                     hors bandes d'outpaint, en niveaux 8 bits :
                       bias = moyenne signee (derive de couleur), max, p999
                       lp   = ecart basse frequence (flou 3 px modele) = derive
                              visible, le detail perdu au resize n'y compte pas
    edge_lp          meme ecart pres du bord de l'image (16 px + rayon du filtre)
                     (liseres : gris des bandes qui deborde dans l'image ; sans
                     bande, le Lanczos seul y laisse ~3 niveaux)
    """
    base = run["base"][0].numpy()
    res = run["result"][0].numpy()
    comb = run["combined"][0].numpy()
    bands = run["bands"][0].numpy() > 0.5         # bandes d'outpaint (a generer)
    H, W = comb.shape
    C = min(base.shape[2], res.shape[2])
    diff = res[..., :C] - base[..., :C]
    changed = np.abs(diff).max(axis=-1) > 0

    union = np.zeros((H, W), bool)
    for r in run["regions"]:
        rr = r[0] if r.dim() == 3 else r
        union |= rr.numpy() >= 0.5
    factor = max([1.0] + [i["orig_width"] / i["width"] for i in run["infos"]]
                 + [i["orig_height"] / i["height"] for i in run["infos"]])
    adj, fea = run["cfg"]["mask_adjust"], run["cfg"]["feather"]
    # resize du masque (Lanczos 3 px modele) + chemin sous-echantillonne du
    # grow / feather au-dela de 24 px (bilineaire : ~1 cellule de plus)
    # deux Lanczos (aller a l'echelle modele, retour a l'echelle crop) : 3 px de
    # chaque cote de l'echelle la plus grossiere, +1 d'arrondi
    fringe = 3 + math.ceil(3 * factor) + 1 + math.ceil(max(0, adj) / 24) + math.ceil(fea / 24)
    # fondu vers l'exterieur (v1.8) : grow ceil(f/2) + flou ceil(f/2) -> <= f + 1
    reach = max(0, adj) + fea + 1 + fringe
    if union.any():
        dist = ndimage.distance_transform_cdt(~union, metric="chessboard")
        spill = float(dist[changed].max()) if changed.any() else 0.0
        beyond = int((changed & (dist > reach)).sum())
    else:
        spill, beyond = 0.0, int(changed.sum())

    inner = (comb >= 0.999) & ~bands
    out = dict(
        outside_changed=int((changed & (comb <= 0)).sum()),
        changed=int(changed.sum()),
        spill_px=spill, reach_px=reach, beyond_reach=beyond,
        inner_px=int(inner.sum()),
    )
    if inner.any():
        d = diff[inner] * LEVEL
        out["inner_bias"] = float(d.mean())
        out["inner_bias_rgb"] = [float(v) for v in d.mean(axis=0)]
        out["inner_max"] = float(np.abs(d).max())
        out["inner_p999"] = float(np.percentile(np.abs(d), 99.9))
        k = 3.0 * factor
        lp = np.abs(_lowpass(diff, k)).max(axis=-1) * LEVEL
        # hors d'atteinte du filtre : bord de la zone recollee et bandes d'outpaint
        margin = int(math.ceil(2 * k)) + 2
        if (~inner).any():
            core = inner & (ndimage.distance_transform_cdt(inner, metric="chessboard") > margin)
        else:
            core = inner.copy()
        if bands.any():
            core &= ndimage.distance_transform_cdt(~bands, metric="chessboard") > margin + 16
        edge = np.zeros((H, W), bool)
        eb = 16 + margin
        edge[:eb, :] = edge[-eb:, :] = True
        edge[:, :eb] = edge[:, -eb:] = True
        # interieur : loin du bord de l'image (le Lanczos de PIL y tronque son
        # noyau, ~3 niveaux meme sans bande) ; le bord est mesure a part
        interior = core & ~edge
        out["inner_lp"] = float(lp[interior].max()) if interior.any() else 0.0
        e = core & edge
        out["edge_lp"] = float(lp[e].max()) if e.any() else 0.0
    return out


def phase_shift(a, b):
    """Translation (dx, dy) de b par rapport a a, correlation de phase (torch.fft), sous-pixel."""
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    h, w = a.shape
    win = torch.outer(torch.hann_window(h, periodic=False, dtype=torch.float64),
                      torch.hann_window(w, periodic=False, dtype=torch.float64))
    Fa, Fb = torch.fft.fft2(a * win), torch.fft.fft2(b * win)
    R = Fb * Fa.conj()
    R /= R.abs().clamp_min(1e-12)
    r = torch.fft.ifft2(R).real
    idx = int(torch.argmax(r))
    py, px = divmod(idx, w)

    def sub(c, n, get):
        l, m, rr = get((c - 1) % n), get(c), get((c + 1) % n)
        den = l - 2 * m + rr
        return c + (0.5 * (l - rr) / den if abs(den) > 1e-12 else 0.0)

    fy = sub(py, h, lambda k: float(r[k, px]))
    fx = sub(px, w, lambda k: float(r[py, k]))
    if fy > h / 2:
        fy -= h
    if fx > w / 2:
        fx -= w
    return fx, fy


def measure_shift(run):
    """
    Pour un modele qui glisse : deplacement mesure dans l'image finale (px) sur
    la zone recollee, et ecart au raccord (anneau du feather) en niveaux 8 bits.
    """
    base = run["base"][0].numpy()[..., :3].mean(-1)
    res = run["result"][0].numpy()[..., :3].mean(-1)
    comb = run["combined"][0].numpy()
    inner = ndimage.binary_erosion(comb >= 0.999, iterations=8)
    if inner.sum() < 64 * 64:
        return None
    ys, xs = np.where(inner)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    dx, dy = phase_shift(base[y0:y1, x0:x1], res[y0:y1, x0:x1])
    ring = (comb > 0.02) & (comb < 0.98)
    seam = np.abs(res - base)[ring] * LEVEL if ring.any() else np.zeros(1)
    info = run["infos"][0]
    return dict(dx=dx, dy=dy,
                factor=info["orig_width"] / info["width"],
                seam_p99=float(np.percentile(seam, 99)), seam_mean=float(seam.mean()))


# ---------------------------------------------------------------------------
# Tirage de configurations
# ---------------------------------------------------------------------------

SIZES = [(512, 512), (640, 480), (480, 640), (1000, 667), (1024, 1024), (1080, 1920),
         (1920, 1080), (1536, 640), (777, 1333), (2000, 1333), (3000, 2000), (333, 251)]
MASKS = ["none", "blob", "edge", "corner", "line", "multi", "tiny", "full"]


def random_config(rng):
    w, h = rng.choice(SIZES)
    return dict(
        size=(w, h),
        mask=rng.choice(MASKS), soft=rng.random() < 0.3,
        scale=rng.choice([1.0, 1.0, 0.5, 0.73, 1.5]),
        ratio=rng.choice(RATIOS),
        regenerate_all=rng.random() < 0.15,
        padding=rng.choice([0, 16, 48, 128]),
        multiple=rng.choice(MULTIPLES),
        resolution=rng.choice([0, 512, 1024, 1152, 1536]),
        force_square=rng.random() < 0.5,
        force_target_downscale=rng.random() < 0.6,
        context=rng.choice(["region", "region", "full_image"]),
        color_fix=rng.random() < 0.4,
        mask_adjust=rng.choice([0, 6, 20, -4]),
        feather=rng.choice([0, 5, 8, 30]),
        order=rng.choice(["list_first_on_top", "area_large_under"]),
        seed=rng.randrange(1 << 30),
    )


def run_config(c, model=model_identity):
    w, h = c["size"]
    src = make_image(w, h, seed=c["seed"])
    mask = make_mask(w, h, c["mask"], seed=c["seed"], soft=c["soft"])
    kw = {k: v for k, v in c.items() if k in DEFAULTS}
    return run_chain(src, mask, model=model, **kw)


def _fmt(c):
    keys = ["size", "mask", "scale", "ratio", "regenerate_all", "multiple", "resolution",
            "force_square", "force_target_downscale", "context", "color_fix",
            "mask_adjust", "feather"]
    return " ".join(f"{k}={c[k]}" for k in keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shift", action="store_true", help="mesure modele qui glisse / zoome")
    args = ap.parse_args()

    if args.shift:
        shift_report()
        return

    rng = random.Random(args.seed)
    worst = {}
    fails = 0
    for i in range(args.n):
        c = random_config(rng)
        m = measure(run_config(c))
        bad = m["outside_changed"] > 0 or m["beyond_reach"] > 0
        if bad:
            fails += 1
            print("ECHEC", _fmt(c), m)
        for k in ("inner_bias", "inner_lp", "edge_lp", "spill_px"):
            v = m.get(k)
            if v is not None and (k not in worst or abs(v) > abs(worst[k][0])):
                worst[k] = (v, _fmt(c))
    print(f"\n{args.n} configs, {fails} echec(s) hors masque")
    for k, (v, c) in worst.items():
        print(f"pire {k:12s} = {v:8.2f}  [{c}]")


def shift_report():
    """Tableau : ce que devient un glissement / zoom du modele dans l'image finale."""
    print("modele          | crop -> modele | facteur | depl. mesure (px) | raccord p99 (niv. 8 bits)")
    cases = [
        ("full + SCALE 1, square", dict(size=(3000, 2000), mask="none", resolution=1024, force_square=True)),
        ("full + SCALE 1, rect", dict(size=(3000, 2000), mask="none", resolution=1024, force_square=False)),
        ("full + SCALE 0.5, rect", dict(size=(3000, 2000), mask="none", scale=0.5, resolution=1024)),
        ("blob, region", dict(size=(2000, 1333), mask="blob", resolution=1024)),
    ]
    for label, c in cases:
        cfg = dict(seed=7, soft=False)
        cfg.update(c)
        for model in (make_model_shift(1.0), make_model_shift(2.0), make_model_zoom(0.005)):
            w, h = cfg["size"]
            src = make_image(w, h, seed=cfg["seed"])
            mask = make_mask(w, h, cfg["mask"], seed=cfg["seed"])
            kw = {k: v for k, v in cfg.items() if k in DEFAULTS}
            run = run_chain(src, mask, model=model, **kw)
            s = measure_shift(run)
            i = run["infos"][0]
            print(f"{label:24s} {model.__name__:14s} | {i['orig_width']}x{i['orig_height']} -> "
                  f"{i['width']}x{i['height']} | x{s['factor']:.2f} | "
                  f"({s['dx']:+.2f}, {s['dy']:+.2f}) | {s['seam_p99']:.1f}")


if __name__ == "__main__":
    main()
