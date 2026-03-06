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

def round_up_8(n):
    return math.ceil(n / 8) * 8


class RatioOutpaintCalc:
    """
    Calcule automatiquement le padding pour outpainter une image
    vers un ratio cible standard. Sort IMAGE + MASK prêts pour
    VAE Encode (Inpaint).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "ratio": (["none"] + list(RATIOS.keys()),),
            }
        }

    RETURN_TYPES  = ("IMAGE", "MASK")
    RETURN_NAMES  = ("image_padded", "mask")
    FUNCTION      = "calculate"
    CATEGORY      = "Aioli Nodes"

    def calculate(self, image, ratio):
        B, H, W, C = image.shape

        # Mode none : pass-through, aucun padding
        if ratio == "none":
            print(f"[RatioOutpaintCalc] {W}x{H} | ratio=none, pass-through")
            mask = torch.zeros((B, H, W), dtype=torch.float32)
            return (image, mask)

        rw, rh = RATIOS[ratio]

        # Comparaison croisée entière (évite les erreurs float)
        if W * rh < H * rw:
            new_W = round_up_8(math.ceil(H * rw / rh))
            new_H = H
        elif W * rh > H * rw:
            new_W = W
            new_H = round_up_8(math.ceil(W * rh / rw))
        else:
            print(f"[RatioOutpaintCalc] {W}x{H} déjà au ratio {ratio}")
            mask = torch.zeros((B, H, W), dtype=torch.float32)
            return (image, mask)

        pad_left   = (new_W - W) // 2
        pad_right  = (new_W - W) - pad_left
        pad_top    = (new_H - H) // 2
        pad_bottom = (new_H - H) - pad_top

        print(f"[RatioOutpaintCalc] {W}x{H} → {new_W}x{new_H} | {ratio} | L={pad_left} R={pad_right} T={pad_top} B={pad_bottom}")

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
