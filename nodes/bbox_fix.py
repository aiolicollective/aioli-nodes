import math
import torch
import numpy as np
from PIL import Image


TARGET_SIZES  = ["none", "512", "768", "1024", "1536", "2048"]
RATIO_OPTIONS = ["free", "1:1"]

# Hard cap for Flux / SDXL compatibility (max pixels on any side) — used only when target="none"
MAX_SIDE = 2048


class BBoxMultipleFix:
    """
    S'insère après 'Mask Bounding Box' (ComfyUI Essentials).

    crop_ratio (nouveau) :
      "free" — comportement par défaut, ratio libre du bbox.
      "1:1"  — force le crop à être carré (côté = max(width, height)),
               en restant aligné sur le multiple choisi.

    Mode none (target="none") :
      Arrondit width/height au multiple choisi.
      Si le crop dépasse MAX_SIDE (2048) : downscale GCD vers MAX_SIDE.

    Mode target (512/768/1024/1536/2048) — UNIFIÉ upscale ET downscale :
      Le même algorithme GCD s'applique que le crop soit plus petit OU plus
      grand que la target :
        1. Calcule t_w × t_h (target × ?) en multiple de mult.
        2. Extrait le ratio irréductible a/b.
        3. Trouve new_w × new_h = a*k × b*k le plus proche du bbox.
           → crop et target ont le MÊME ratio exact → pixel-perfect.
        4. Resize Lanczos : crop → target (upscale ou downscale).

    Sorties :
      image_cropped / mask_cropped  → VAE Encode (Inpaint)
      x / y                         → ImageCompositeMasked
      orig_width / orig_height      → dimensions crop DANS la source
      width / height                → dimensions finales (après resize)
      target_size                   → valeur INT du target (0 si "none")
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":      ("IMAGE",),
                "mask":       ("MASK",),
                "x":          ("INT", {"default": 0,  "min": 0, "max": 99999}),
                "y":          ("INT", {"default": 0,  "min": 0, "max": 99999}),
                "width":      ("INT", {"default": 64, "min": 1, "max": 99999}),
                "height":     ("INT", {"default": 64, "min": 1, "max": 99999}),
                "multiple":   (["8 (VAE minimum)", "32 (SD1.5)", "64 (SDXL / Flux)"],),
                "target":     (TARGET_SIZES,  {"default": "none"}),
                "crop_ratio": (RATIO_OPTIONS, {"default": "free"}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "MASK", "INT", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES  = ("image_cropped", "mask_cropped",
                     "x", "y",
                     "orig_width", "orig_height",
                     "width", "height",
                     "target_size")
    FUNCTION      = "fix"
    CATEGORY      = "Aioli Nodes"

    def fix(self, image, mask, x, y, width, height, multiple, target, crop_ratio="free"):

        mult = int(multiple.split(" ")[0])
        B, H_src, W_src, C = image.shape

        # ── Step 1 : ratio forcé sur le bbox de base ─────────────────────────
        if crop_ratio == "1:1":
            base = max(width, height)
            base_w, base_h = base, base
        else:
            base_w, base_h = width, height

        # ── Step 2 : calcul new_w / new_h / up_w / up_h ──────────────────────
        need_resize  = False
        resize_label = ""
        up_w = up_h  = 0

        if target != "none":
            t = int(target)

            # Dimensions cible en multiple de mult, ratio du crop de base
            if base_w >= base_h:
                t_w = t
                t_h = math.ceil((base_h * t / base_w) / mult) * mult
            else:
                t_h = t
                t_w = math.ceil((base_w * t / base_h) / mult) * mult

            # GCD crop adjustment — MÊME logique pour upscale ET downscale
            g    = math.gcd(t_w, t_h)
            a, b = t_w // g, t_h // g
            k    = round((base_w / a + base_h / b) / 2)
            k    = max(1, k)
            new_w, new_h = a * k, b * k
            up_w, up_h   = t_w, t_h

            if t_w != new_w or t_h != new_h:
                need_resize  = True
                resize_label = "upscale " if t_w > new_w else "downscale"

        else:
            # Mode none : arrondi au multiple, cap MAX_SIDE si dépassement
            new_w = math.ceil(base_w / mult) * mult
            new_h = math.ceil(base_h / mult) * mult

            if new_w > MAX_SIDE or new_h > MAX_SIDE:
                # Downscale GCD vers MAX_SIDE (ratio exact préservé)
                if new_w >= new_h:
                    down_w = MAX_SIDE
                    down_h = math.ceil((new_h * MAX_SIDE / new_w) / mult) * mult
                else:
                    down_h = MAX_SIDE
                    down_w = math.ceil((new_w * MAX_SIDE / new_h) / mult) * mult

                g    = math.gcd(down_w, down_h)
                a, b = down_w // g, down_h // g
                k    = round((new_w / a + new_h / b) / 2)
                k    = max(1, k)
                new_w, new_h = a * k, b * k
                up_w, up_h   = down_w, down_h
                need_resize  = True
                resize_label = "downscale"
            else:
                up_w, up_h = new_w, new_h

        # ── Step 3 : expansion symétrique autour du bbox original ─────────────
        # On centre le crop (new_w × new_h) sur le bbox d'origine (x, y, width, height)
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

        ratio_info = f"  ratio={crop_ratio}" if crop_ratio != "free" else ""
        print(f"[BBoxMultipleFix] bbox     : {width}x{height} @({x},{y}){ratio_info}")
        print(f"[BBoxMultipleFix] crop     : {orig_w}x{orig_h} @({new_x},{new_y})")

        # ── Step 4 : crop ─────────────────────────────────────────────────────
        img_cropped = image[:, new_y:new_y + new_h, new_x:new_x + new_w, :]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        mask_cropped = mask[:, new_y:new_y + new_h, new_x:new_x + new_w]

        # ── Step 5 : resize Lanczos si nécessaire ────────────────────────────
        if need_resize and (up_w != orig_w or up_h != orig_h):
            print(f"[BBoxMultipleFix] {resize_label} : {orig_w}x{orig_h} → {up_w}x{up_h}")
            img_cropped  = self._resize(img_cropped,  up_w, up_h, "image")
            mask_cropped = self._resize(mask_cropped, up_w, up_h, "mask")
            final_w, final_h = up_w, up_h
        else:
            final_w, final_h = orig_w, orig_h

        target_size = 0 if target == "none" else int(target)

        return (img_cropped, mask_cropped, new_x, new_y, orig_w, orig_h, final_w, final_h, target_size)

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
