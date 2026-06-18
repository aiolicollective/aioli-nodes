import torch


class RegionalMaskConditioning:
    """
    Replie une LISTE de (conditioning, masque) en UN conditioning regional pour
    une generation en UNE passe KSampler (mode "tiles=1" + DyPE). Chaque paire
    (caption_i, masque_i) devient un 'Conditioning (Set Mask)', le tout concatene.

    >>> COUCHE GLOBALE OPTIONNELLE (v1.4) <<<
    'base_conditioning' = un prompt qui decrit l'IMAGE ENTIERE. Il est applique
    sur toute la surface (masque plein) et MELANGE aux regions via 'base_strength'
    (= son mask_strength). En zone : le pixel recoit un melange pondere
    base_strength (global) vs strength (region). Hors region (si pas de fond) :
    le global seul. base_strength=0 OU base non branche -> regional pur, identique
    a avant. Le melange se fait au niveau du sampler (mask_strength) -> pas de lerp
    de tenseurs, compatible avec des prompts de longueurs differentes.

    Cout : le global ajoute +1 forward pass / step (1 region de plus). Pour la
    vitesse, 'set_area_to_bounds' limite chaque region a sa bbox.

    BROADCAST : 1 seul conditioning pour N masques -> applique a toutes les regions.
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
                "base_conditioning":  ("CONDITIONING",),
                "base_strength":      ("FLOAT",   {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),
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

    def combine(self, conditioning, masks, strength=None, set_area_to_bounds=None,
                base_conditioning=None, base_strength=None):
        strength           = float((strength or [1.0])[0])
        set_area_to_bounds = bool((set_area_to_bounds or [False])[0])
        base_strength      = float((base_strength or [0.5])[0])

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

        # couche globale optionnelle (prompt image entiere), melangee via base_strength
        base_added = 0
        if base_conditioning and base_strength > 0.0 and covered is not None:
            base_flat = [t for c in base_conditioning for t in c]
            if base_flat:
                H, W = covered.shape[-2], covered.shape[-1]
                full = torch.ones((H, W), dtype=torch.float32)
                result = self._set_mask(base_flat, full, base_strength, False) + result
                base_added = len(base_flat)

        print(f"[RegionalMaskConditioning] {n} regions"
              f"{' (broadcast 1->N)' if broadcast else ''}"
              f"{f' + base global x{base_strength}' if base_added else ''}"
              f" -> {len(result)} tokens")
        return (result, covered.unsqueeze(0))
