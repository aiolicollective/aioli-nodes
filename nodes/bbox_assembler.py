import torch
import numpy as np
from PIL import Image


def _resize_img(hwc, new_w, new_h):
    arr = (hwc.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(arr).resize((new_w, new_h), Image.LANCZOS)
    return torch.from_numpy(np.array(pil).astype(np.float32) / 255.0)


def _resize_mask(hw, new_w, new_h):
    arr = (hw.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(arr, mode='L').resize((new_w, new_h), Image.LANCZOS)
    return torch.from_numpy(np.array(pil).astype(np.float32) / 255.0)


def _feather(mask_hw, px):
    if px <= 0:
        return mask_hw
    m = mask_hw.unsqueeze(0).unsqueeze(0)
    k = int(px) * 2 + 1
    m = torch.nn.functional.avg_pool2d(m, kernel_size=k, stride=1, padding=int(px))
    return m.squeeze(0).squeeze(0).clamp(0.0, 1.0)


class BBoxMultipleAssembler:
    """
    Recompose pixel-perfect d'une LISTE de crops sur une seule image de base.

    Pendant 'assemblage' du flux SAM3-regional / multi-bbox : chaque masque est
    recadre (Mask Bounding Box / BBox Multiple Fix) puis enhance separement dans
    le KSampler. Ce node replie la liste des N crops samples sur le fond, a leurs
    coordonnees d'origine.

    Resout trois choses que ComfyUI ne sait pas faire nativement :
      * N crops (liste) -> 1 image : fold cumulatif sur une seule toile.
      * Arrondi /16 du sampler : chaque crop est resize a (width, height) de sa
        bbox source AVANT collage -> zero drift, meme si le KSampler a renvoye
        une taille legerement differente.
      * Chevauchement : les grandes zones sont collees d'abord (dessous), les
        petites ensuite (dessus), avec feather sur le masque pour des bords doux.

    INPUT_IS_LIST = True -> toutes les entrees arrivent en listes. Les singletons
    (base_image, feather, opacity, larger_underneath) sont pris en element [0].

    S'apparie avec 'Mask Bounding Box' (Essentials) ou 'BBox Multiple Fix' :
      x / y            -> position du crop dans la source
      width / height   -> dimensions du crop dans la source (orig_width/orig_height)
      masks            -> masque de l'objet (forme libre) pour le blending
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
                "feather":           ("INT",     {"default": 0,   "min": 0, "max": 256}),
                "opacity":           ("FLOAT",   {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "larger_underneath": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "combined_mask")
    FUNCTION     = "assemble"
    CATEGORY     = "Aioli Nodes"

    def assemble(self, base_image, crops, masks, x, y, width, height,
                 feather=None, opacity=None, larger_underneath=None):

        # ── unwrap singletons (INPUT_IS_LIST emballe tout en listes) ──────────
        base = base_image[0]
        if base.dim() == 4:
            base = base[0]
        base = base.clone().float()
        H, W, _ = base.shape

        feather  = int((feather  or [0])[0])
        opacity  = float((opacity or [1.0])[0])
        larger_underneath = bool((larger_underneath or [True])[0])

        n = min(len(crops), len(masks), len(x), len(y), len(width), len(height))
        combined = torch.zeros((H, W), dtype=torch.float32)

        # ── ordre de collage : grandes bbox dessous, petites dessus ───────────
        order = list(range(n))
        if larger_underneath:
            order.sort(key=lambda i: int(width[i]) * int(height[i]), reverse=True)

        pasted = 0
        for i in order:
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

            # ── snap crop + masque a la taille bbox source (corrige le /16) ───
            if crop.shape[0] != hi or crop.shape[1] != wi:
                crop = _resize_img(crop, wi, hi)
            if m.shape[0] != hi or m.shape[1] != wi:
                m = _resize_mask(m, wi, hi)

            a = _feather(m, feather) * opacity

            # ── clip a la toile (gere les bbox en bord d'image) ──────────────
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
            pasted += 1

        print(f"[BBoxMultipleAssembler] assemblage : {pasted}/{n} crops sur toile {W}x{H}")
        return (base.clamp(0.0, 1.0).unsqueeze(0), combined.unsqueeze(0))
