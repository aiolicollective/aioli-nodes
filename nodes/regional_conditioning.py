import torch


class RegionalMaskConditioning:
    """
    Replie une LISTE de (conditioning, masque) en UN seul conditioning regional,
    pour une generation en UNE seule passe KSampler (mode "tiles = 1" + DyPE).

    Modele mental : c'est le regional facon TTP, sauf que les "tuiles" sont les
    masques SAM3 (+ le masque de fond inverse, deja dans la liste si tu l'as
    active dans RegionMaskList). Chaque paire (caption_i, masque_i) devient un
    'Conditioning (Set Mask)' ; le tout est concatene. ComfyUI ne sait pas
    combiner une liste dynamique de N -> ce node le fait pour un N quelconque.

    Le FOND n'a plus de traitement special : il est juste le dernier masque de
    la liste (l'inverse de l'union), avec son propre caption Gemma. Plus simple.

    BROADCAST : si tu ne fournis qu'UN seul conditioning pour N masques, il est
    applique a toutes les regions (pratique pour debug / prompt unique).

    INPUT_IS_LIST = True.
    """

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "masks":        ("MASK",),
            },
            "optional": {
                "strength":           ("FLOAT",   {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_area_to_bounds": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "MASK")
    RETURN_NAMES = ("conditioning", "covered_mask")
    FUNCTION     = "combine"
    CATEGORY     = "Aioli Nodes"

    @staticmethod
    def _set_mask(cond, mask, strength, set_area_to_bounds):
        if mask.dim() < 3:
            mask = mask.unsqueeze(0)
        out = []
        for t in cond:
            d = t[1].copy()
            d["mask"] = mask
            d["set_area_to_bounds"] = bool(set_area_to_bounds)
            d["mask_strength"] = float(strength)
            out.append([t[0], d])
        return out

    def combine(self, conditioning, masks, strength=None, set_area_to_bounds=None):
        strength           = float((strength or [1.0])[0])
        set_area_to_bounds = bool((set_area_to_bounds or [False])[0])

        nC, nM = len(conditioning), len(masks)
        if nC == 0 or nM == 0:
            flat = [t for c in conditioning for t in c]
            return (flat, torch.zeros((1, 1, 1)))

        broadcast = (nC == 1 and nM > 1)
        n = nM if broadcast else min(nC, nM)

        result = []
        covered = None
        for i in range(n):
            m = masks[i]
            if m.dim() == 3:
                m = m[0]
            mm = m.float()
            covered = mm if covered is None else torch.maximum(covered, mm)
            cond_i = conditioning[0] if broadcast else conditioning[i]
            result += self._set_mask(cond_i, m, strength, set_area_to_bounds)

        print(f"[RegionalMaskConditioning] {n} regions"
              f"{' (broadcast 1->N)' if broadcast else ''} -> {len(result)} tokens")
        return (result, covered.unsqueeze(0))
