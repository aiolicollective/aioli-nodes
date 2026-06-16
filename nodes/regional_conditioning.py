import torch


class RegionalMaskConditioning:
    """
    Replie une LISTE de conditionings + une LISTE de masques en UN seul
    conditioning regional, pour une generation en UNE seule passe KSampler.

    Equivalent natif : 'Conditioning (Set Mask)' applique a chaque paire, suivi
    de N 'Conditioning (Combine)'. Sauf que ComfyUI ne sait pas combiner une
    liste de N elements dynamiques -> ce node le fait pour un N quelconque.

    Usage typique (flux SAM3-regional, mode passe unique) :
      SAM3 (N masques) ─┬─ crops -> Gemma -> N captions -> CLIPTextEncode (N cond)
                        └─ masks (pleine taille) ────────────────────┐
      CLIPTextEncode (liste N) ─> conditioning ─┐                                │
                                                ▼                                ▼
                              RegionalMaskConditioning  <─ masks (liste N) ──────┘
                                                │
                                                ▼  (1 seule passe)
                                            KSampler

    'background' (optionnel) : un conditioning applique LA OU aucun masque n'est
    actif (union inversee des masques). Pratique pour decrire le fond.

    INPUT_IS_LIST = True -> conditioning et masks arrivent en listes ; les
    singletons (strength, set_area_to_bounds, background) sont pris en [0].
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
                "background":         ("CONDITIONING",),
                "strength":           ("FLOAT",   {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_area_to_bounds": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "MASK")
    RETURN_NAMES = ("conditioning", "background_mask")
    FUNCTION     = "combine"
    CATEGORY     = "Aioli Nodes"

    @staticmethod
    def _set_mask(cond, mask, strength, set_area_to_bounds):
        # Reproduit la semantique de ConditioningSetMask (ComfyUI core).
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

    def combine(self, conditioning, masks, background=None,
                strength=None, set_area_to_bounds=None):

        strength = float((strength or [1.0])[0])
        set_area_to_bounds = bool((set_area_to_bounds or [False])[0])

        n = min(len(conditioning), len(masks))
        result = []
        union = None

        for i in range(n):
            m = masks[i]
            if m.dim() == 3:
                m = m[0]
            mm = m.float()
            union = mm if union is None else torch.maximum(union, mm)
            result += self._set_mask(conditioning[i], m, strength, set_area_to_bounds)

        # ── fond : union inversee des masques ─────────────────────────────
        if union is None:
            # pas de masque -> renvoie le conditioning brut
            bg_mask = torch.zeros((1, 1))
            flat = [t for c in conditioning for t in c]
            print("[RegionalMaskConditioning] aucun masque -> passthrough")
            return (flat, bg_mask.unsqueeze(0) if bg_mask.dim() == 2 else bg_mask)

        bg_mask = (1.0 - union).clamp(0.0, 1.0)

        if background is not None and len(background) > 0:
            result += self._set_mask(background[0], bg_mask, strength, set_area_to_bounds)

        print(f"[RegionalMaskConditioning] {n} zones combinees"
              f"{' + fond' if background is not None and len(background) > 0 else ''}"
              f" -> {len(result)} tokens de conditioning")
        return (result, bg_mask.unsqueeze(0))
