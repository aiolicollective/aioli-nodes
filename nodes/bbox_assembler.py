import torch
import torch.nn.functional as F
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


def _sep_maxpool(m, px):
    """Dilatation (grow) separable -> O(H*W*k) au lieu de O(H*W*k^2)."""
    k = int(px) * 2 + 1
    m = F.max_pool2d(m, (1, k), stride=1, padding=(0, int(px)))
    m = F.max_pool2d(m, (k, 1), stride=1, padding=(int(px), 0))
    return m


def _grow_shrink_fast(mask_hw, px):
    """
    Grow (px>0) / shrink (px<0) sur la PLEINE TOILE, borne par l'image.
    Rapide meme pour de grandes valeurs : au-dela de 24 px on travaille sur un
    masque sous-echantillonne (le rayon effectif reste petit), puis on re-upscale.
    """
    if px == 0:
        return mask_hw
    px = int(px)
    a = abs(px)
    m = mask_hw.unsqueeze(0).unsqueeze(0).float()

    # downscale pour les gros rayons : kernel effectif plafonne ~24
    scale = 1
    if a > 24:
        scale = int(np.ceil(a / 24.0))
        H, W = m.shape[-2:]
        m = F.avg_pool2d(m, scale, stride=scale, ceil_mode=True)
    r = max(1, round(a / scale))

    if px > 0:
        m = _sep_maxpool(m, r)
    else:
        m = 1.0 - _sep_maxpool(1.0 - m, r)

    if scale > 1:
        m = F.interpolate(m, size=mask_hw.shape, mode="bilinear", align_corners=False)
    return m.squeeze(0).squeeze(0).clamp(0.0, 1.0)


def _feather_fast(mask_hw, px):
    if px <= 0:
        return mask_hw
    px = int(px)
    m = mask_hw.unsqueeze(0).unsqueeze(0).float()
    scale = 1
    if px > 24:
        scale = int(np.ceil(px / 24.0))
        m = F.avg_pool2d(m, scale, stride=scale, ceil_mode=True)
    r = max(1, round(px / scale))
    k = r * 2 + 1
    m = F.avg_pool2d(m, (1, k), stride=1, padding=(0, r))
    m = F.avg_pool2d(m, (k, 1), stride=1, padding=(r, 0))
    if scale > 1:
        m = F.interpolate(m, size=mask_hw.shape, mode="bilinear", align_corners=False)
    return m.squeeze(0).squeeze(0).clamp(0.0, 1.0)


class BBoxMultipleAssembler:
    """
    Recompose pixel-perfect d'une LISTE de crops sur une seule image de base.

    Chaque masque (SAM3 / multi-bbox) est recadre puis enhance separement dans le
    KSampler ; ce node replie la liste des N crops a leurs coordonnees, sur le fond.
    INPUT_IS_LIST = True (singletons pris en [0]).

    >>> EXTENSION SUR PLEINE TOILE (v1.2) <<<
    'mask_adjust' n'est PLUS limite par le crop SAM3. Le masque de chaque crop est
    d'abord POSE sur une toile pleine image (noir partout ailleurs), puis grossi /
    retreci -> l'extension est bornee par l'IMAGE entiere, pas par le crop. On peut
    donc etendre une zone APRES coup sans relancer le KSampler. Le compositing se
    fait UNIQUEMENT dans le rectangle du crop : la ou le masque blanc deborde du
    crop, ce calque n'ecrit rien -> on voit le CALQUE DU DESSOUS (transparence),
    jamais l'image de base imposee. Grow/feather separables + downscale -> rapide
    meme a grande valeur (fini le "temps fou").

    HIERARCHIE ('order') :
      * 'list_first_on_top' : le 1er masque de la liste est AU-DESSUS ; le DERNIER
        est dessous. -> mets le masque de FOND (inverse de l'union) en DERNIER.
      * 'area_large_under'  : grandes zones dessous, petites dessus (auto).

    'checker' : contour colore de chaque zone pour verifier la superposition
    (toujours dessine ; non expose).
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
                "order":       (["list_first_on_top", "area_large_under"],),
                "mask_adjust": ("INT",   {"default": 0,   "min": -512, "max": 512}),
                "feather":     ("INT",   {"default": 8,   "min": 0, "max": 512}),
                "opacity":     ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "combined_mask", "checker")
    FUNCTION     = "assemble"
    CATEGORY     = "Aioli Nodes"

    def assemble(self, base_image, crops, masks, x, y, width, height,
                 order=None, mask_adjust=None, feather=None, opacity=None):

        base = base_image[0]
        if base.dim() == 4:
            base = base[0]
        base = base.clone().float()
        H, W, C = base.shape

        order         = (order or ["list_first_on_top"])[0]
        mask_adjust   = int((mask_adjust or [0])[0])
        feather       = int((feather or [8])[0])
        opacity       = float((opacity or [1.0])[0])
        debug_outline = True  # fige : le checker est toujours annote (sinon == image)

        n = min(len(crops), len(masks), len(x), len(y), len(width), len(height))
        canvas   = base.clone()
        combined = torch.zeros((H, W), dtype=torch.float32)

        idx = list(range(n))
        if order == "area_large_under":
            idx.sort(key=lambda i: int(width[i]) * int(height[i]), reverse=True)
        else:  # list_first_on_top -> on peint du dernier au premier
            idx = list(reversed(idx))

        outlines = []
        for i in idx:
            wi, hi = int(width[i]), int(height[i])
            xi, yi = int(x[i]),     int(y[i])

            crop = crops[i]
            if crop.dim() == 4:
                crop = crop[0]
            crop = crop.float()
            if crop.shape[2] != C:
                crop = crop[..., :C] if crop.shape[2] > C else crop.repeat(1, 1, C)[..., :C]

            m = masks[i]
            if m.dim() == 3:
                m = m[0]
            m = m.float()

            if crop.shape[0] != hi or crop.shape[1] != wi:
                crop = _resize_img(crop, wi, hi)
            if m.shape[0] != hi or m.shape[1] != wi:
                m = _resize_mask(m, wi, hi)

            # rectangle du crop reellement present dans l'image (= la seule zone
            # ou ce calque possede des pixels enhances)
            x0, y0 = max(xi, 0), max(yi, 0)
            x1, y1 = min(xi + wi, W), min(yi + hi, H)
            if x1 <= x0 or y1 <= y0:
                continue
            cx0, cy0 = x0 - xi, y0 - yi
            cx1, cy1 = cx0 + (x1 - x0), cy0 + (y1 - y0)

            # 1) masque pose sur PLEINE TOILE (noir hors crop), grossi/retreci
            #    -> borne par l'IMAGE, jamais clip au bord du crop (fini le "bug")
            full_mask = torch.zeros((H, W), dtype=torch.float32)
            full_mask[y0:y1, x0:x1] = m[cy0:cy1, cx0:cx1]
            full_mask = _grow_shrink_fast(full_mask, mask_adjust)
            alpha_full = _feather_fast(full_mask, feather) * opacity

            # 2) COMPOSITE UNIQUEMENT dans le rectangle du crop.
            #    Hors du crop : on n'ecrit rien -> le calque DESSOUS (hierarchie)
            #    reste visible = "transparence". Pas de pixels de base imposes.
            a   = alpha_full[y0:y1, x0:x1].unsqueeze(-1)
            src = crop[cy0:cy1, cx0:cx1, :]
            dst = canvas[y0:y1, x0:x1, :]
            canvas[y0:y1, x0:x1, :] = src * a + dst * (1.0 - a)
            combined[y0:y1, x0:x1] = torch.maximum(combined[y0:y1, x0:x1],
                                                   alpha_full[y0:y1, x0:x1])

            if debug_outline:
                outlines.append((alpha_full, x0, y0, x1, y1, i))

        if debug_outline:
            checker = canvas.clone()
            for (af, x0, y0, x1, y1, i) in outlines:
                m_rect = torch.zeros_like(af)
                m_rect[y0:y1, x0:x1] = (af[y0:y1, x0:x1] > 0.5).float()
                edge = (_grow_shrink_fast(m_rect, 2) - m_rect).clamp(0, 1).unsqueeze(-1)
                col  = torch.tensor(_PALETTE[i % len(_PALETTE)], dtype=torch.float32)
                checker = checker * (1.0 - edge) + col * edge
        else:
            checker = canvas

        print(f"[BBoxMultipleAssembler] {n} calques, order={order}, mask_adjust={mask_adjust} "
              f"(pleine toile), feather={feather} -> {W}x{H}")
        return (canvas.clamp(0.0, 1.0).unsqueeze(0),
                combined.unsqueeze(0),
                checker.clamp(0.0, 1.0).unsqueeze(0))
