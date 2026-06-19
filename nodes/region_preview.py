import torch

from .bbox_assembler import _grow_shrink_fast, _feather_fast, _resize_mask, _PALETTE


class RegionPreview:
    """
    Pré-visualisation EN AMONT du rendu de BBoxMultipleAssembler.

    À brancher juste après Region Mask List ou Mask Split Regions (branche
    latérale : ne touche pas au pipeline liste). Prend l'image de base + la LISTE
    des masques et restitue, AVANT toute génération :
      - combined_mask : union des masques (mask_adjust / feather / opacity
                        appliqués EXACTEMENT comme l'assembler) ;
      - checker       : l'image de base + contour & léger remplissage colorés par
                        zone, pour voir sur quoi on travaille.

    Sert aussi de POINT DE RÉGLAGE UNIQUE : règle order / mask_adjust / feather /
    opacity ici, visualise l'effet, et rebranche les sorties dans
    BBoxMultipleAssembler (convertis ses widgets en entrées). La preview = le rendu
    final car on réutilise la maths de l'assembler (helpers importés).

    Masques pleine-taille (sortie habituelle de Region Mask List / Mask Split
    Regions) : aucun coord requis. Si tu fournis x/y/width/height (BBoxMultipleFix),
    les masques sont traités comme des crops et placés à leurs coordonnées.

    INPUT_IS_LIST = True (comme l'assembler ; singletons pris en [0]).
    """

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "masks":      ("MASK",),
            },
            "optional": {
                "x":             ("INT", {"forceInput": True}),
                "y":             ("INT", {"forceInput": True}),
                "width":         ("INT", {"forceInput": True}),
                "height":        ("INT", {"forceInput": True}),
                "order":         (["list_first_on_top", "area_large_under"],),
                "mask_adjust":   ("INT",     {"default": 0,   "min": -512, "max": 512}),
                "feather":       ("INT",     {"default": 8,   "min": 0,    "max": 512}),
                "opacity":       ("FLOAT",   {"default": 1.0, "min": 0.0,  "max": 1.0, "step": 0.01}),
                "debug_outline": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("combined_mask", "checker", "mask_adjust", "feather", "opacity", "order")
    FUNCTION     = "preview"
    CATEGORY     = "Aioli Nodes"

    def preview(self, base_image, masks, x=None, y=None, width=None, height=None,
                order=None, mask_adjust=None, feather=None, opacity=None, debug_outline=None):

        base = base_image[0]
        if base.dim() == 4:
            base = base[0]
        base = base.clone().float()
        H, W, C = base.shape

        order         = (order or ["list_first_on_top"])[0]
        mask_adjust   = int((mask_adjust or [0])[0])
        feather       = int((feather or [8])[0])
        opacity       = float((opacity or [1.0])[0])
        debug_outline = bool((debug_outline or [True])[0])

        n = len(masks)
        has_coords = bool(x and y and width and height
                          and len(x) >= n and len(y) >= n
                          and len(width) >= n and len(height) >= n)

        combined = torch.zeros((H, W), dtype=torch.float32)
        layers = []
        for i in range(n):
            m = masks[i]
            if m.dim() == 3:
                m = m[0]
            m = m.float()

            if has_coords:
                wi, hi = int(width[i]), int(height[i])
                xi, yi = int(x[i]),     int(y[i])
                if m.shape[0] != hi or m.shape[1] != wi:
                    m = _resize_mask(m, wi, hi)
                x0, y0 = max(xi, 0), max(yi, 0)
                x1, y1 = min(xi + wi, W), min(yi + hi, H)
                if x1 <= x0 or y1 <= y0:
                    continue
                cx0, cy0 = x0 - xi, y0 - yi
                cx1, cy1 = cx0 + (x1 - x0), cy0 + (y1 - y0)
                full_mask = torch.zeros((H, W), dtype=torch.float32)
                full_mask[y0:y1, x0:x1] = m[cy0:cy1, cx0:cx1]
            else:
                if m.shape[0] != H or m.shape[1] != W:
                    m = _resize_mask(m, W, H)
                full_mask = m

            full_mask = _grow_shrink_fast(full_mask, mask_adjust)
            alpha = (_feather_fast(full_mask, feather) * opacity).clamp(0.0, 1.0)
            combined = torch.maximum(combined, alpha)
            layers.append((alpha, i))

        checker = base.clone()
        if debug_outline:
            for (alpha, i) in layers:
                col  = torch.tensor(_PALETTE[i % len(_PALETTE)], dtype=torch.float32)
                fill = (alpha * 0.35).unsqueeze(-1)
                checker = checker * (1.0 - fill) + col * fill
                hard = (alpha > 0.5).float()
                edge = (_grow_shrink_fast(hard, 2) - hard).clamp(0.0, 1.0).unsqueeze(-1)
                checker = checker * (1.0 - edge) + col * edge

        print(f"[RegionPreview] {n} zones, coords={'oui' if has_coords else 'non'}, "
              f"mask_adjust={mask_adjust}, feather={feather}, opacity={opacity}")
        return (combined.unsqueeze(0),
                checker.clamp(0.0, 1.0).unsqueeze(0),
                mask_adjust, feather, opacity, order)
