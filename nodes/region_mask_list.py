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

    Cette meme liste alimente : MaskBoundingBox+ (coords + crop par region),
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
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "INT")
    RETURN_NAMES = ("masks", "background_mask", "count")
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION = "build"
    CATEGORY = "Aioli Nodes"

    def build(self, masks, add_background=None, threshold=None, min_bg_area=None):
        add_background = bool((add_background or [True])[0])
        threshold      = float((threshold or [0.5])[0])
        min_bg_area    = float((min_bg_area or [0.0])[0])

        # aplatit batch + liste -> liste de masques HxW
        flat = []
        for entry in masks:
            t = entry
            if t.dim() == 3:           # batch B,H,W
                for k in range(t.shape[0]):
                    flat.append(t[k].float())
            elif t.dim() == 2:
                flat.append(t.float())
            else:
                flat.append(t.reshape(t.shape[-2], t.shape[-1]).float())

        if not flat:
            empty = torch.zeros((1, 1), dtype=torch.float32)
            return ([empty], empty.unsqueeze(0), 0)

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

        print(f"[RegionMaskList] {len(flat)} masques SAM3"
              f"{' + fond (inverse union)' if len(out) > len(flat) else ''} -> {len(out)} calques")
        return (out, bg.unsqueeze(0), len(out))
