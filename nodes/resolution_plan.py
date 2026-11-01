"""
Plan de resolution : ce que la chaine des wf V2 va faire de l'image, calcule
avec les MEMES fonctions que les nodes (une seule source de verite) :

    ImageScaleBy (SCALE) -> Ratio Outpaint Calc (multiple = none)
      -> Mask Split Regions -> Region Mask List (+ bandes ou image entiere)
      -> Aioli Mask BBox (bbox, crop, taille modele, bandes)

Sans dependance a ComfyUI (testable hors ComfyUI). Utilise par le panneau
Resolution (route serveur pour l'apercu avant run, et execution du node).
"""

import math

import torch
import torch.nn.functional as F

from .bbox_core_v2 import MAX_RESOLUTION, MULTIPLES, compute_crop_v2
from .log import quiet
from .mask_bbox_v2 import CONTEXTS, pad_bbox, tight_bbox
from .mask_split_regions import MaskSplitRegions
from .ratio_outpaint import RATIOS, RatioOutpaintCalc, outpaint_layout
from .region_mask_list import RegionMaskList

IMAGE_ENTIERE = "whole image"
RATIO_OPTIONS = ["none"] + list(RATIOS.keys())
MAX_PREVIEW_PIXELS = 40_000_000     # apercu avec masque : au-dela, regions au run

DEFAULTS = dict(scale=1.0, ratio="none", regenerate_all=False, resolution=1024,
                multiple="16 (Flux)", force_square=False, force_target_downscale=True,
                padding=48, context="region")


def scaled_size(W, H, scale):
    """Taille apres ImageScaleBy (arrondi de ComfyUI)."""
    if scale == 1.0:
        return W, H
    return round(W * scale), round(H * scale)


def region_layout(src_w, src_h, mask=None, scale=1.0, ratio="none", regenerate_all=False):
    """
    Partie lente (masques) : taille apres SCALE et outpaint, et bbox serre de
    chaque region, dans l'ordre de la liste de Region Mask List.
    mask : MASK (B, H, W) a la taille source ou None. Le resultat ne depend que
    de ces arguments : il peut etre mis en cache pendant qu'on regle le reste.
    """
    sw, sh = scaled_size(src_w, src_h, scale)
    lay = outpaint_layout(sw, sh, ratio, "none")
    if lay is None:
        PW, PH, pads = sw, sh, (0, 0, 0, 0)
    else:
        PW, PH = lay[0], lay[1]
        pads = tuple(lay[2:6])          # gauche, droite, haut, bas

    pl, pr, pt, pb = pads
    if mask is None:
        # Sans masque : Mask Split Regions ne rend rien, Region Mask List garde
        # la region d'extra (bandes ou image entiere) ou, a defaut, l'image
        # entiere. Calcul direct, sans tenseur pleine taille (teste contre la
        # chaine dans tests/test_resolution_plan.py).
        if regenerate_all or not any(pads):
            out = [dict(kind=IMAGE_ENTIERE, tight=(0, PW, 0, PH))]
        else:
            x0, x1 = (0, PW) if (pt or pb) else ((0 if pl else pl + sw), (PW if pr else pl))
            y0, y1 = (0, PH) if (pl or pr) else ((0 if pt else pt + sh), (PH if pb else pt))
            out = [dict(kind="outpaint", tight=(x0, x1, y0, y1))]
        return dict(source=(src_w, src_h), scaled=(sw, sh), padded=(PW, PH), pads=pads,
                    regions=out)

    with torch.no_grad(), quiet():
        m = mask.float()
        if m.dim() == 2:
            m = m.unsqueeze(0)
        in_mask = RatioOutpaintCalc._input_mask(m, m.shape[0], sh, sw)
        mask_padded = F.pad(in_mask, pads, mode="constant", value=0.0)
        split, _ = MaskSplitRegions().split(mask_padded, empty_mask="nothing")
        if regenerate_all:
            extra, extra_kind = torch.ones((1, PH, PW)), IMAGE_ENTIERE
        else:
            extra = F.pad(torch.zeros((1, sh, sw)), pads, mode="constant", value=1.0)
            extra_kind = "outpaint"
        regions, _, _ = RegionMaskList().build(list(split), add_background=[False],
                                               threshold=[0.5], min_bg_area=[0.0],
                                               extra_mask=[extra], empty=["full_image"])

    n_drawn = sum(1 for r in split if (r >= 0.5).any())
    out = []
    for k, r in enumerate(regions):
        if k < n_drawn:
            kind = "mask"
        elif k == n_drawn and (extra >= 0.5).any():
            kind = extra_kind
        else:
            kind = IMAGE_ENTIERE           # aucun masque : toute l'image est editee
        out.append(dict(kind=kind, tight=tight_bbox(r)))
    return dict(source=(src_w, src_h), scaled=(sw, sh), padded=(PW, PH), pads=pads,
                regions=out)


def plan_from_layout(lay, resolution=1024, multiple="16 (Flux)", force_square=False,
                     force_target_downscale=True, padding=48, context="region", **_):
    """Partie rapide : bbox, crop, taille modele de chaque region (Aioli Mask BBox)."""
    PW, PH = lay["padded"]
    regions = []
    for k, r in enumerate(lay["regions"]):
        with quiet():
            x, y, w, h = pad_bbox(r["tight"], padding, context, PW, PH)
            c = compute_crop_v2(x, y, w, h, PW, PH, multiple, resolution,
                                force_square, force_target_downscale)
        cw, ch = c["orig_width"], c["orig_height"]
        l, t, rr, b = c["bands"]
        band_area = cw * ch - (cw - l - rr) * (ch - t - b)
        regions.append(dict(
            index=k + 1, kind=r["kind"],
            bbox=[x, y, w, h],
            crop=[c["x"], c["y"], cw, ch],
            model=[c["width"], c["height"]],
            bands=[l, t, rr, b],
            band_pct=round(100.0 * band_area / (cw * ch), 1),
            resize=c["resize_label"],
            # >1 : crop agrandi pour le modele ; <1 : reduit. Un glissement de
            # 1 px du modele devient 1 / model_scale px dans l'image finale.
            model_scale=round(c["width"] / cw, 3),
        ))
    sw, sh = lay["scaled"]
    return dict(source=list(lay["source"]), scaled=[sw, sh], padded=[PW, PH],
                pads=list(lay["pads"]),
                coverage=round(100.0 * sw * sh / (PW * PH), 1),
                regions=regions, warnings=_warnings(lay, regions, sw, sh, PW, PH))


def plan(src_w, src_h, mask=None, **settings):
    s = dict(DEFAULTS)
    s.update({k: v for k, v in settings.items() if k in DEFAULTS})
    lay = region_layout(src_w, src_h, mask, s["scale"], s["ratio"], s["regenerate_all"])
    p = plan_from_layout(lay, **s)
    p["settings"] = s
    return p


def _warnings(lay, regions, sw, sh, PW, PH):
    w = []
    cov = 100.0 * sw * sh / (PW * PH)
    if (PW, PH) != (sw, sh) and cov < 45:
        w.append(f"outpaint: the original covers only {cov:.0f} % of the canvas")
    for r in regions:
        if r["model_scale"] < 0.5:
            w.append(f"R{r['index']}: crop shrunk x{1 / r['model_scale']:.1f} for the model "
                     f"(detail lost; a 1 px slide of the model = {1 / r['model_scale']:.1f} px in the result)")
        if r["band_pct"] >= 25:
            w.append(f"R{r['index']}: {r['band_pct']:.0f} % grey bands in the crop")
    return w


def model_summary(p):
    """Taille modele reelle des regions (pas MODEL SIZE^2 : plafond, SHRINK off...)."""
    mps = [r["model"][0] * r["model"][1] / 1e6 for r in p["regions"]]
    if not mps:
        return "-"
    lo, hi = min(mps), max(mps)
    return f"{hi:.1f} MP" if hi - lo < 0.05 else f"{lo:.1f}–{hi:.1f} MP per region"


def report(p):
    """Texte lisible (sortie STRING du panneau, pour PreviewAny)."""
    s = p["settings"]
    sw, sh = p["scaled"]
    PW, PH = p["padded"]
    lines = [f"source {p['source'][0]}x{p['source'][1]}"
             + (f" -> OUTPUT SCALE x{s['scale']:g} = {sw}x{sh}" if [sw, sh] != p["source"] else "")
             + (f" -> outpaint {s['ratio']} = {PW}x{PH} (original {p['coverage']:.0f} %)"
                if [PW, PH] != [sw, sh] else ""),
             f"output {PW}x{PH} - model {model_summary(p)}"]
    for r in p["regions"]:
        x, y, w, h = r["bbox"]
        cx, cy, cw, ch = r["crop"]
        mw, mh = r["model"]
        sc = r["model_scale"]
        size = ("no resize" if not r["resize"] else
                f"{r['resize']} x{sc:.2f}" if sc >= 1 else f"{r['resize']} x{sc:.2f} (1/{1 / sc:.1f})")
        lines.append(f"R{r['index']} {r['kind']}: bbox {w}x{h} @({x},{y}) -> crop {cw}x{ch} @({cx},{cy})"
                     f" -> model {mw}x{mh}, {size}"
                     + (f", bands {r['band_pct']:.0f} %" if r["band_pct"] else ""))
    lines += [f"! {x}" for x in p["warnings"]]
    return "\n".join(lines)


def validate_settings(raw):
    """
    Reglages recus du navigateur -> dict valide, bornes comme les widgets du
    panneau. ValueError (message lisible) si une valeur sort des plages.
    """
    s = dict(DEFAULTS)
    raw = {} if raw is None else raw
    if not isinstance(raw, dict):
        raise ValueError("settings: object expected")

    def num(key, kind, lo, hi):
        v = raw.get(key, s[key])
        try:
            finite = not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)
        except OverflowError:           # entier JSON geant
            finite = False
        if not finite:
            raise ValueError(f"{key}: number expected")
        v = kind(v)
        if not (lo <= v <= hi):
            raise ValueError(f"{key} out of range ({lo}-{hi}): {v}")
        return v

    def choice(key, options):
        v = raw.get(key, s[key])
        if v not in options:
            raise ValueError(f"{key}: unknown value {v!r}")
        return v

    def flag(key):
        v = raw.get(key, s[key])
        if not isinstance(v, bool):
            raise ValueError(f"{key}: boolean expected")
        return v

    s["scale"] = num("scale", float, 0.01, 8.0)
    s["resolution"] = num("resolution", int, 0, MAX_RESOLUTION)
    s["padding"] = num("padding", int, 0, 4096)
    s["ratio"] = choice("ratio", RATIO_OPTIONS)
    s["multiple"] = choice("multiple", MULTIPLES)
    s["context"] = choice("context", CONTEXTS)
    for k in ("regenerate_all", "force_square", "force_target_downscale"):
        s[k] = flag(k)
    return s
