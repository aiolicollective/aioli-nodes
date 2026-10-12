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
    AioliMaskBBox se mappe dessus automatiquement.

    Pour une seule tache -> 1 masque (comportement identique a avant).

    Widgets exposes (les seuls qu'on regle vraiment) :
      threshold : binarisation du masque dessine (0..1)
      min_area  : taches plus petites (en pixels) ignorees (anti-points perdus)
      sort_by   : ordre des regions en sortie (= hierarchie de l'assembler)
      empty_mask: que faire d'un masque vide (v1.7.0)
                  nothing    -> 1 region a zeros, count = 0 (comportement d'avant)
                  full_image -> 1 region pleine (toute l'image), count = 1 :
                                pas de masque = toute l'image est editee.
                  "Vide" est juge APRES threshold : un masque peint a moins de
                  threshold partout compte comme vide.

    Reglages figes dans le code (rarement touches, defauts qui conviennent a la
    quasi-totalite des masques dessines ; modifiables ici au besoin) :
      connectivity = 8   : les diagonales comptent. Mettre 4 pour ne relier que
                           les pixels cote a cote (taches qui se touchent par un
                           coin restent separees).
      merge_distance = 0 : pas de fusion. Mettre > 0 pour dilater avant le
                           decoupage et fusionner des traits voisins en une
                           seule region.

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
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_area":  ("INT",   {"default": 64, "min": 0, "max": 4096 * 4096}),
                "sort_by":   (["area_desc", "area_asc", "top_to_bottom", "left_to_right"],),
                "empty_mask": (["nothing", "full_image"], {"default": "nothing"}),
            },
        }

    RETURN_TYPES = ("MASK", "INT")
    RETURN_NAMES = ("masks", "count")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "split"
    CATEGORY = "Aioli Nodes"

    # connectivity / merge_distance : defauts figes (non exposes en widget)
    def split(self, mask, threshold=0.5, min_area=64, sort_by="area_desc",
              empty_mask="nothing", merge_distance=0, connectivity=8):
        # MASK comfy : (H,W) ou (B,H,W) -> on prend le 1er
        m = mask
        if m.dim() == 3:
            m = m[0]
        soft = m.float().cpu()
        H, W = soft.shape
        binary = (soft.numpy() >= float(threshold))

        if not binary.any():
            if empty_mask == "full_image":
                print("[MaskSplitRegions] masque vide -> 1 region = image entiere (empty_mask=full_image)")
                return ([torch.ones((H, W), dtype=torch.float32)], 1)
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
