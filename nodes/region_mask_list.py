import torch


class RegionMaskList:
    """
    Prepare la LISTE de masques regionaux pour le pipeline SAM3.

    Entree : les N masques SAM3 (batch OU liste). Sortie : une LISTE propre de
    masques pleine-taille, dans l'ordre recu (= ordre de priorite donne par le
    prompt Gemma : du + important/grand au + petit), avec en OPTION le masque de
    FOND ajoute EN DERNIER (= inverse de l'union de tous les masques).

    Le DERNIER element doit etre le fond : avec BBoxMultipleAssembler en
    'list_first_on_top' et RegionalMaskConditioning, "dernier = dessous" -> le
    fond reste sous les objets. C'est pour ca que add_background ajoute en queue.

    v1.7.1, deux options (non branchees / par defaut : comportement d'avant) :
      extra_mask : un masque ajoute TEL QUEL comme une region de plus, apres
                   les autres (avant le fond), s'il n'est pas vide.
                   Typiquement le masque des bandes
                   de Ratio Outpaint Calc : une seule region, toute l'image en
                   contexte, a cote des masques dessines ou SAM3.
      empty      : keep (defaut) -> les masques vides passent comme avant.
                   full_image   -> les masques vides (rien >= threshold) sont
                   retires ; s'il ne reste aucune region, la liste devient UN
                   masque plein : pas de masque = toute l'image est editee.

    Cette meme liste alimente : AioliMaskBBox V2 (bbox + crop + coords par region),
    Gemma (caption par region), le KSampler (N passes) ET le conditionnement
    regional mode B. Le fond y est donc lui aussi captionne et enhance.

    INPUT_IS_LIST = True.
    """

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
            },
            "optional": {
                "add_background": ("BOOLEAN", {"default": True}),
                "threshold":      ("FLOAT",   {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_bg_area":    ("FLOAT",   {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "extra_mask":     ("MASK",),
                "empty":          (["keep", "full_image"], {"default": "keep"}),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "INT")
    RETURN_NAMES = ("masks", "background_mask", "count")
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION = "build"
    CATEGORY = "Aioli Nodes"

    @staticmethod
    def _flatten(masks):
        """batch + liste -> liste de masques HxW."""
        flat = []
        for entry in masks or []:
            if entry is None:
                continue
            t = entry
            if t.dim() == 3:           # batch B,H,W
                for k in range(t.shape[0]):
                    flat.append(t[k].float())
            elif t.dim() == 2:
                flat.append(t.float())
            else:
                flat.append(t.reshape(t.shape[-2], t.shape[-1]).float())
        return flat

    def build(self, masks, add_background=None, threshold=None, min_bg_area=None,
              extra_mask=None, empty=None):
        add_background = bool((add_background or [True])[0])
        threshold      = float((threshold or [0.5])[0])
        min_bg_area    = float((min_bg_area or [0.0])[0])
        empty          = (empty or ["keep"])[0]

        flat = self._flatten(masks)
        extras = self._flatten(extra_mask)
        n_in, n_extra = len(flat), len(extras)

        ref = flat[0] if flat else (extras[0] if extras else None)
        # un extra_mask vide (ratio d'outpaint = none) n'ajoute jamais de region
        extras = [m for m in extras if (m >= threshold).any()]
        if empty == "full_image":
            flat = [m for m in flat if (m >= threshold).any()]
        flat = flat + extras

        if not flat:
            if empty == "full_image" and ref is not None:
                full = torch.ones_like(ref)
                print(f"[RegionMaskList] aucune region ({n_in} masques, {n_extra} extra, vides) "
                      f"-> 1 region = image entiere (empty=full_image)")
                return ([full], torch.zeros_like(ref).unsqueeze(0), 1)
            empty_m = torch.zeros((1, 1), dtype=torch.float32) if ref is None else torch.zeros_like(ref)
            return ([empty_m], empty_m.unsqueeze(0), 0)

        H, W = flat[0].shape
        union = torch.zeros((H, W), dtype=torch.float32)
        for m in flat:
            if m.shape != (H, W):
                m = torch.nn.functional.interpolate(
                    m.unsqueeze(0).unsqueeze(0), size=(H, W),
                    mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
            union = torch.maximum(union, (m >= threshold).float())

        bg = (1.0 - union).clamp(0.0, 1.0)

        out = list(flat)
        if add_background and bg.mean().item() > min_bg_area:
            out.append(bg)

        extra_info = f" (dont {len(extras)} extra)" if extras else ""
        print(f"[RegionMaskList] {len(flat)} masques{extra_info}"
              f"{' + fond (inverse union)' if len(out) > len(flat) else ''} -> {len(out)} calques")
        return (out, bg.unsqueeze(0), len(out))
