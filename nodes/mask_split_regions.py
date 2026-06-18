import torch
import numpy as np

try:
    from scipy import ndimage as _ndi          # deja livre avec ComfyUI
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _binary_dilate(mask_bool, px):
    """Dilatation binaire separable rapide (sans scipy), pour merge_distance."""
    if px <= 0:
        return mask_bool
    import torch.nn.functional as F
    m = torch.from_numpy(mask_bool.astype(np.float32))[None, None]
    k = int(px) * 2 + 1
    m = F.max_pool2d(m, (1, k), stride=1, padding=(0, int(px)))
    m = F.max_pool2d(m, (k, 1), stride=1, padding=(int(px), 0))
    return (m[0, 0].numpy() > 0.5)


class MaskSplitRegions:
    """
    Decoupe UN masque (dessine a la main) en N masques separes, un par tache
    deconnectee (connected components). Chaque region peut ensuite avoir sa
    propre bbox / son propre crop / son propre prompt -> inpaint multiple a la
    main, exactement comme SAM3 mais sans mots-clefs.

    Sortie = LISTE de N masques pleine-taille (la tache en place, noir ailleurs),
    dans l'ordre choisi. Branche-la la ou tu branchais le masque manuel unique :
    MaskBoundingBox+ / BBoxMultipleFix se mappent dessus automatiquement.

    Pour une seule tache -> 1 masque (comportement identique a avant).

    Parametres :
      threshold      : binarisation du masque dessine (0..1)
      min_area       : taches plus petites (en pixels) ignorees (anti-points perdus)
      merge_distance : rapproche/fusionne les traits proches avant decoupe (0 = strict)
      connectivity   : 4 (cote a cote) ou 8 (diagonales comptent)
      sort_by        : ordre des regions en sortie

    Dependance : scipy.ndimage (deja livre avec ComfyUI). Aucun install requis.
    INPUT_IS_LIST = False (un seul masque en entree, on mappe pas).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            },
            "optional": {
                "threshold":      ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_area":       ("INT",   {"default": 64, "min": 0, "max": 4096 * 4096}),
                "merge_distance": ("INT",   {"default": 0, "min": 0, "max": 256}),
                "connectivity":   ([8, 4],),
                "sort_by":        (["area_desc", "area_asc", "top_to_bottom", "left_to_right"],),
            },
        }

    RETURN_TYPES = ("MASK", "INT")
    RETURN_NAMES = ("masks", "count")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "split"
    CATEGORY = "Aioli Nodes"

    def split(self, mask, threshold=0.5, min_area=64, merge_distance=0,
              connectivity=8, sort_by="area_desc"):
        # MASK comfy : (H,W) ou (B,H,W) -> on prend le 1er
        m = mask
        if m.dim() == 3:
            m = m[0]
        soft = m.float().cpu()
        H, W = soft.shape
        binary = (soft.numpy() >= float(threshold))

        if not binary.any():
            print("[MaskSplitRegions] masque vide -> 1 region (zeros)")
            return ([torch.zeros((H, W), dtype=torch.float32)], 0)

        # zone de labelling (eventuellement fusionnee)
        lab_input = _binary_dilate(binary, merge_distance) if merge_distance > 0 else binary

        if _HAS_SCIPY:
            if int(connectivity) == 8:
                structure = np.ones((3, 3), dtype=np.int32)
            else:
                structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int32)
            labels, n = _ndi.label(lab_input, structure=structure)
        else:
            print("[MaskSplitRegions] scipy introuvable -> masque non decoupe (1 region)")
            return ([soft], 1)

        # construit un masque doux par composante (garde les valeurs dessinees)
        soft_np = soft.numpy()
        regions = []
        for i in range(1, n + 1):
            comp = (labels == i)
            area = int(comp.sum())
            if area < int(min_area):
                continue
            reg = soft_np * comp.astype(np.float32)      # valeurs douces, hors tache = 0
            regions.append((reg, area, comp))

        if not regions:
            print(f"[MaskSplitRegions] {n} taches mais toutes < min_area -> 1 region (union)")
            return ([soft], 1)

        # tri / hierarchie
        def topmost(comp):
            ys, xs = np.where(comp)
            return (ys.min(), xs.min())
        if sort_by == "area_desc":
            regions.sort(key=lambda r: r[1], reverse=True)
        elif sort_by == "area_asc":
            regions.sort(key=lambda r: r[1])
        elif sort_by == "top_to_bottom":
            regions.sort(key=lambda r: topmost(r[2])[0])
        elif sort_by == "left_to_right":
            regions.sort(key=lambda r: topmost(r[2])[1])

        out = [torch.from_numpy(r[0]).float() for r in regions]
        print(f"[MaskSplitRegions] {n} taches detectees -> {len(out)} regions "
              f"(min_area={min_area}, merge={merge_distance}, conn={connectivity})")
        return (out, len(out))
