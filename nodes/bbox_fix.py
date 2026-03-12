import math
import torch
import numpy as np
from PIL import Image


TARGET_SIZES = ["none", "512", "768", "1024", "1536", "2048"]

# Hard cap for Flux / SDXL compatibility (max pixels on any side)
MAX_SIDE = 2048


class BBoxMultipleFix:
    """
    S'insère après 'Mask Bounding Box' (ComfyUI Essentials).

    Mode none :
      Arrondit width/height au multiple choisi (×8/32/64).
      Crop reste au plus proche du bbox d'origine.
      Le crop est plafonné à MAX_SIDE (2048) pour la compatibilité Flux.

    Mode target (512/768/1024/1536/2048) :
      1. Calcule les dimensions upscale (target × ?) en multiple de 64.
      2. Extrait le ratio irréductible a/b de l'upscale.
      3. Trouve crop_w × crop_h = a*k × b*k le plus proche du bbox.
         → crop et upscale ont le MÊME ratio exact → mask parfaitement
           aligné, downscale retour pixel-perfect.
      4. Upscale Lanczos : crop → target.
      Si target ≤ bbox (pas de sens d'upscaler) : fallback → mode none
      avec cap à MAX_SIDE pour rester compatible Flux.

    Sorties :
      image_cropped / mask_cropped  → VAE Encode (Inpaint)
      x / y                         → ImageCompositeMasked
      orig_width / orig_height      → dimensions crop AVANT upscale,
                                      pour resize retour après VAE Decode
      width / height                → dimensions finales (après upscale)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":    ("IMAGE",),
                "mask":     ("MASK",),
                "x":        ("INT", {"default": 0,  "min": 0, "max": 99999}),
                "y":        ("INT", {"default": 0,  "min": 0, "max": 99999}),
                "width":    ("INT", {"default": 64, "min": 1, "max": 99999}),
                "height":   ("INT", {"default": 64, "min": 1, "max": 99999}),
                "multiple": (["8 (VAE minimum)", "32 (SD1.5)", "64 (SDXL / Flux)"],),
                "target":   (TARGET_SIZES, {"default": "none"}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "MASK", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES  = ("image_cropped", "mask_cropped",
                     "x", "y",
                     "orig_width", "orig_height",
                     "width", "height")
    FUNCTION      = "fix"
    CATEGORY      = "Aioli Nodes"

    def fix(self, image, mask, x, y, width, height, multiple, target):

        mult = int(multiple.split(" ")[0])
        B, H_src, W_src, C = image.shape

        use_upscale = False

        if target != "none":
            t = int(target)

            # Dimensions upscale en multiple de 64
            if width >= height:
                up_w = t
                up_h = math.ceil((height * t / width) / 64) * 64
            else:
                up_h = t
                up_w = math.ceil((width * t / height) / 64) * 64

            # Fallback si target ≤ bbox (inutile d'upscaler)
            if up_w > width and up_h > height:
                use_upscale = True
                g = math.gcd(up_w, up_h)
                a, b = up_w // g, up_h // g
                k = round((width / a + height / b) / 2)
                k = max(1, k)
                new_w = a * k
                new_h = b * k
            else:
                print(f"[BBoxMultipleFix] target={t} ≤ bbox → fallback mode none (cap {MAX_SIDE}px)")

        if not use_upscale:
            # Arrondi au multiple supérieur
            new_w = math.ceil(width  / mult) * mult
            new_h = math.ceil(height / mult) * mult

            # ── FIX : cap à MAX_SIDE pour la compatibilité Flux ──────────────
            if new_w > MAX_SIDE or new_h > MAX_SIDE:
                scale = MAX_SIDE / max(new_w, new_h)
                # floor pour ne pas dépasser MAX_SIDE, puis alignement multiple
                new_w = max(mult, math.floor(new_w * scale / mult) * mult)
                new_h = max(mult, math.floor(new_h * scale / mult) * mult)
                print(f"[BBoxMultipleFix] capped   : crop réduit à {new_w}x{new_h} (max {MAX_SIDE}px/côté)")
            # ─────────────────────────────────────────────────────────────────

            up_w = new_w
            up_h = new_h

        # Expansion symétrique autour du bbox
        new_x = x - (new_w - width)  // 2
        new_y = y - (new_h - height) // 2

        # Clamp dans les bords de l'image source
        new_x = max(0, new_x)
        new_y = max(0, new_y)
        if new_x + new_w > W_src: new_x = W_src - new_w
        if new_y + new_h > H_src: new_y = H_src - new_h
        new_w = min(new_w, W_src)
        new_h = min(new_h, H_src)
        new_x = max(0, new_x)
        new_y = max(0, new_y)

        orig_w, orig_h = new_w, new_h

        print(f"[BBoxMultipleFix] bbox     : {width}x{height} @({x},{y})")
        print(f"[BBoxMultipleFix] crop     : {orig_w}x{orig_h} @({new_x},{new_y})")

        # Crop image et mask
        img_cropped = image[:, new_y:new_y + new_h, new_x:new_x + new_w, :]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        mask_cropped = mask[:, new_y:new_y + new_h, new_x:new_x + new_w]

        # Upscale Lanczos si target valide
        if use_upscale and (up_w != orig_w or up_h != orig_h):
            print(f"[BBoxMultipleFix] upscale  : {orig_w}x{orig_h} → {up_w}x{up_h}")
            img_cropped  = self._resize(img_cropped,  up_w, up_h, "image")
            mask_cropped = self._resize(mask_cropped, up_w, up_h, "mask")
            final_w, final_h = up_w, up_h
        else:
            final_w, final_h = orig_w, orig_h

        return (img_cropped, mask_cropped, new_x, new_y, orig_w, orig_h, final_w, final_h)

    def _resize(self, tensor, new_W, new_H, mode):
        frames = []
        for b in range(tensor.shape[0]):
            arr = (tensor[b].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            if mode == "image":
                pil = Image.fromarray(arr).resize((new_W, new_H), Image.LANCZOS)
            else:
                pil = Image.fromarray(arr, mode='L').resize((new_W, new_H), Image.LANCZOS)
            frames.append(np.array(pil).astype(np.float32) / 255.0)
        return torch.from_numpy(np.stack(frames))
