import torch
import numpy as np
from PIL import Image

_PALETTE = [
    (1.0, 0.2, 0.2), (0.2, 0.8, 1.0), (0.4, 1.0, 0.3), (1.0, 0.85, 0.1),
    (0.9, 0.3, 1.0), (1.0, 0.55, 0.0), (0.2, 1.0, 0.8), (1.0, 0.4, 0.7),
]


def _resize_img(hwc, new_w, new_h):
    arr = (hwc.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(arr).resize((new_w, new_h), Image.LANCZOS)
    return torch.from_numpy(np.array(pil).astype(np.float32) / 255.0)


def _resize_mask(hw, new_w, new_h):
    arr = (hw.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(arr, mode='L').resize((new_w, new_h), Image.LANCZOS)
    return torch.from_numpy(np.array(pil).astype(np.float32) / 255.0)


def _grow_shrink(mask_hw, px):
    if px == 0:
        return mask_hw
    m = mask_hw.unsqueeze(0).unsqueeze(0)
    k = int(abs(px)) * 2 + 1
    if px > 0:
        m = torch.nn.functional.max_pool2d(m, k, stride=1, padding=int(abs(px)))
    else:
        m = 1.0 - torch.nn.functional.max_pool2d(1.0 - m, k, stride=1, padding=int(abs(px)))
    return m.squeeze(0).squeeze(0).clamp(0.0, 1.0)


def _feather(mask_hw, px):
    if px <= 0:
        return mask_hw
    m = mask_hw.unsqueeze(0).unsqueeze(0)
    k = int(px) * 2 + 1
    m = torch.nn.functional.avg_pool2d(m, k, stride=1, padding=int(px))
    return m.squeeze(0).squeeze(0).clamp(0.0, 1.0)


class BBoxMultipleAssembler:
    """
    Recompose pixel-perfect d'une LISTE de crops sur une seule image de base.

    Chaque masque (SAM3 / multi-bbox) est recadre puis enhance separement dans
    le KSampler ; ce node replie la liste des N crops a leurs coordonnees, sur le
    fond. INPUT_IS_LIST = True (singletons pris en [0]).

    HIERARCHIE DE RECOUVREMENT (parametre 'order') :
      * 'list_first_on_top'  -> le 1er masque de la liste est AU-DESSUS des autres
        (colle en ordre inverse : le dernier en premier, donc dessous). Pratique
        avec SAM3 ou le 1er mot du prompt a la priorite, et ou la derniere entree
        peut etre le masque inverse du fond.
      * 'area_large_under'   -> grandes zones dessous, petites dessus (auto).

    'mask_adjust' (grow +/ shrink -) redefinit les zones blanches du masque AVANT
    le feather -> controle fin du recouvrement. 'feather' adoucit les bords.

    Sortie 'checker' : l'image avec le contour de chaque zone trace en couleur,
    pour verifier d'un coup d'oeil que tout se superpose bien.
    """

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "crops":      ("IMAGE",),
                "masks":      ("MASK",),
                "x":          ("INT", {"forceInput": True}),
                "y":          ("INT", {"forceInput": True}),
                "width":      ("INT", {"forceInput": True}),
                "height":     ("INT", {"forceInput": True}),
            },
            "optional": {
                "order":         (["list_first_on_top", "area_large_under"],),
                "feather":       ("INT",     {"default": 8,   "min": 0, "max": 256}),
                "mask_adjust":   ("INT",     {"default": 0,   "min": -128, "max": 128}),
                "opacity":       ("FLOAT",   {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "debug_outline": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "combined_mask", "checker")
    FUNCTION     = "assemble"
    CATEGORY     = "Aioli Nodes"

    def assemble(self, base_image, crops, masks, x, y, width, height,
                 order=None, feather=None, mask_adjust=None, opacity=None, debug_outline=None):

        base = base_image[0]
        if base.dim() == 4:
            base = base[0]
        base = base.clone().float()
        H, W, _ = base.shape

        order        = (order or ["list_first_on_top"])[0]
        feather      = int((feather or [8])[0])
        mask_adjust  = int((mask_adjust or [0])[0])
        opacity      = float((opacity or [1.0])[0])
        debug_outline = bool((debug_outline or [True])[0])

        n = min(len(crops), len(masks), len(x), len(y), len(width), len(height))
        combined = torch.zeros((H, W), dtype=torch.float32)

        idx = list(range(n))
        if order == "area_large_under":
            idx.sort(key=lambda i: int(width[i]) * int(height[i]), reverse=True)
        else:  # list_first_on_top -> colle du dernier au premier
            idx = list(reversed(idx))

        outlines = []
        for i in idx:
            wi, hi = int(width[i]), int(height[i])
            xi, yi = int(x[i]), int(y[i])

            crop = crops[i]
            if crop.dim() == 4:
                crop = crop[0]
            crop = crop.float()

            m = masks[i]
            if m.dim() == 3:
                m = m[0]
            m = m.float()

            if crop.shape[0] != hi or crop.shape[1] != wi:
                crop = _resize_img(crop, wi, hi)
            if m.shape[0] != hi or m.shape[1] != wi:
                m = _resize_mask(m, wi, hi)

            m = _grow_shrink(m, mask_adjust)
            a = _feather(m, feather) * opacity

            x0, y0 = max(xi, 0), max(yi, 0)
            x1, y1 = min(xi + wi, W), min(yi + hi, H)
            if x1 <= x0 or y1 <= y0:
                continue
            cx0, cy0 = x0 - xi, y0 - yi
            cx1, cy1 = cx0 + (x1 - x0), cy0 + (y1 - y0)

            dst = base[y0:y1, x0:x1, :]
            src = crop[cy0:cy1, cx0:cx1, :]
            av  = a[cy0:cy1, cx0:cx1].unsqueeze(-1)
            base[y0:y1, x0:x1, :] = src * av + dst * (1.0 - av)
            combined[y0:y1, x0:x1] = torch.maximum(combined[y0:y1, x0:x1], a[cy0:cy1, cx0:cx1])

            if debug_outline:
                outlines.append((m.clamp(0, 1), x0, y0, cx0, cy0, x1 - x0, y1 - y0, i))

        if debug_outline:
            checker = base.clone()
            for (mfull, x0, y0, cx0, cy0, ww, hh, i) in outlines:
                edge = (_grow_shrink(mfull, 2) - mfull)[cy0:cy0 + hh, cx0:cx0 + ww].clamp(0, 1)
                col = torch.tensor(_PALETTE[i % len(_PALETTE)], dtype=torch.float32)
                e = edge.unsqueeze(-1)
                reg = checker[y0:y0 + hh, x0:x0 + ww, :]
                checker[y0:y0 + hh, x0:x0 + ww, :] = reg * (1.0 - e) + col * e
        else:
            checker = base

        print(f"[BBoxMultipleAssembler] {n} crops, order={order}, feather={feather}, "
              f"mask_adjust={mask_adjust} -> toile {W}x{H}")
        return (base.clamp(0.0, 1.0).unsqueeze(0), combined.unsqueeze(0), checker.clamp(0.0, 1.0).unsqueeze(0))
