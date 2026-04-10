import math
import torch


RATIOS = {
    "1:1":  (1, 1),
    "4:5":  (4, 5),
    "5:4":  (5, 4),
    "3:4":  (3, 4),
    "4:3":  (4, 3),
    "16:9": (16, 9),
    "9:16": (9, 16),
}

MULTIPLES = ["none", "8 (VAE minimum)", "16 (Flux)", "32 (SD1.5)", "64 (SDXL)"]


def round_up_8(n):
    """Comportement original : plafond au multiple de 8 superieur."""
    return math.ceil(n / 8) * 8


def snap_to_mult(n, mult):
    """Arrondi au multiple le plus proche (minimise l'ecart de ratio)."""
    return max(mult, round(n / mult) * mult)


class RatioOutpaintCalc:
    """
    Calcule automatiquement le padding pour outpainter une image
    vers un ratio cible standard. Sort IMAGE + MASK prets pour
    VAE Encode (Inpaint).

    multiple :
      none              - comportement original : ceil au multiple de 8
                         uniquement sur la dimension calculee (legacy).
      8 / 16 / 32 / 64 - arrondi au multiple le plus proche (round)
                         sur les deux dimensions.
                         Recommande : 16 (Flux), 64 (SDXL).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":    ("IMAGE",),
                "ratio":    (["none"] + list(RATIOS.keys()),),
                "multiple": (MULTIPLES, {"default": "16 (Flux)"}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "MASK")
    RETURN_NAMES  = ("image_padded", "mask")
    FUNCTION      = "calculate"
    CATEGORY      = "Aioli Nodes"

    def calculate(self, image, ratio, multiple="16 (Flux)"):
        B, H, W, C = image.shape

        # Mode ratio=none : pass-through, aucun padding
        if ratio == "none":
            print(f"[RatioOutpaintCalc] {W}x{H} | ratio=none, pass-through")
            mask = torch.zeros((B, H, W), dtype=torch.float32)
            return (image, mask)

        rw, rh = RATIOS[ratio]

        # --- Calcul des nouvelles dimensions ---
        if multiple == "none":
            # Comportement legacy : ceil x8 sur la dimension calculee uniquement
            if W * rh < H * rw:
                new_W = round_up_8(math.ceil(H * rw / rh))
                new_H = H
            elif W * rh > H * rw:
                new_W = W
                new_H = round_up_8(math.ceil(W * rh / rw))
            else:
                print(f"[RatioOutpaintCalc] {W}x{H} deja au ratio {ratio}")
                mask = torch.zeros((B, H, W), dtype=torch.float32)
                return (image, mask)
            mult_label = "legacy x8-ceil"

        else:
            mult = int(multiple.split(" ")[0])
            if W * rh < H * rw:
                # Image trop etroite : elargir W
                new_W = snap_to_mult(H * rw / rh, mult)
                new_H = snap_to_mult(H, mult)
            elif W * rh > H * rw:
                # Image trop courte : elargir H
                new_W = snap_to_mult(W, mult)
                new_H = snap_to_mult(W * rh / rw, mult)
            else:
                print(f"[RatioOutpaintCalc] {W}x{H} deja au ratio {ratio}")
                mask = torch.zeros((B, H, W), dtype=torch.float32)
                return (image, mask)
            mult_label = f"x{mult}-round"

        pad_left   = (new_W - W) // 2
        pad_right  = (new_W - W) - pad_left
        pad_top    = (new_H - H) // 2
        pad_bottom = (new_H - H) - pad_top

        print(f"[RatioOutpaintCalc] {W}x{H} -> {new_W}x{new_H} | {ratio} | {mult_label} | L={pad_left} R={pad_right} T={pad_top} B={pad_bottom}")

        img = image.permute(0, 3, 1, 2)
        img = torch.nn.functional.pad(
            img,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0.5,
        )
        img = img.permute(0, 2, 3, 1)

        mask = torch.ones((B, new_H, new_W), dtype=torch.float32)
        mask[:, pad_top:pad_top + H, pad_left:pad_left + W] = 0.0

        return (img, mask)
