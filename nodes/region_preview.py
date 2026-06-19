import torch

from .bbox_assembler import _grow_shrink_fast, _feather_fast, _resize_mask, _PALETTE


class RegionPreview:
    """
    Pré-visualisation EN AMONT du rendu de BBoxMultipleAssembler.

    À brancher juste après Region Mask List ou Mask Split Regions (branche
    latérale : ne touche pas au pipeline liste). Prend l'image de base + la LISTE
    des masques pleine-taille et restitue, AVANT toute génération :
      - combined_mask : union des masques (mask_adjust / feather / opacity
                        appliqués EXACTEMENT comme l'assembler) ;
      - checker       : l'image de base + contour & léger remplissage colorés par
                        zone, pour voir sur quoi on travaille.

    Sert aussi de POINT DE RÉGLAGE : règle mask_adjust / feather / opacity ici,
    visualise l'effet, et rebranche ces sorties (INT / INT / FLOAT) dans
    BBoxMultipleAssembler (convertis ses widgets en entrées). 'order' n'est PAS
    exposé ici : c'est un combo, et ComfyUI ne permet pas de relier une sortie de
    node vers une entrée combo — règle-le directement sur l'assembler (il ne change
    pas la preview, c'est juste l'ordre de superposition au compositing final).
    La preview = le rendu final car on réutilise la maths de l'assembler.

    Le contour de debug est TOUJOURS dessiné (figé, non exposé) : sans lui le
    checker n'afficherait que l'image de base nue.

    Les masques attendus sont pleine-taille (sortie habituelle de Region Mask List
    / Mask Split Regions), donc aucune coordonnée n'est nécessaire.

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
                "mask_adjust": ("INT",   {"default": 0,   "min": -512, "max": 512}),
                "feather":     ("INT",   {"default": 8,   "min": 0,    "max": 512}),
                "opacity":     ("FLOAT", {"default": 1.0, "min": 0.0,  "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("combined_mask", "checker", "mask_adjust", "feather", "opacity")
    FUNCTION     = "preview"
    CATEGORY     = "Aioli Nodes"

    def preview(self, base_image, masks,
                mask_adjust=None, feather=None, opacity=None):

        base = base_image[0]
        if base.dim() == 4:
            base = base[0]
        base = base.clone().float()
        H, W, C = base.shape

        mask_adjust = int((mask_adjust or [0])[0])
        feather     = int((feather or [8])[0])
        opacity     = float((opacity or [1.0])[0])
        debug_outline = True  # figé : sans le contour le checker n'afficherait rien

        n = len(masks)
        combined = torch.zeros((H, W), dtype=torch.float32)
        layers = []
        for i in range(n):
            m = masks[i]
            if m.dim() == 3:
                m = m[0]
            m = m.float()
            if m.shape[0] != H or m.shape[1] != W:
                m = _resize_mask(m, W, H)

            full_mask = _grow_shrink_fast(m, mask_adjust)
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

        print(f"[RegionPreview] {n} zones, "
              f"mask_adjust={mask_adjust}, feather={feather}, opacity={opacity}")
        return (combined.unsqueeze(0),
                checker.clamp(0.0, 1.0).unsqueeze(0),
                mask_adjust, feather, opacity)
