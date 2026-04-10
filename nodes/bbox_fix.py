import math
import torch
import numpy as np
from PIL import Image


TARGET_SIZES = ["none", "512", "768", "1024", "1536", "2048"]

# Hard cap for Flux / SDXL compatibility — used only in none mode when force_target_downscale=False
MAX_SIDE = 2048


class BBoxMultipleFix:
    """
    S'insere apres 'Mask Bounding Box' (ComfyUI Essentials).

    force_square (bool) :
      False - comportement par defaut, ratio libre du bbox.
      True  - force le crop a etre carre (cote = max(width, height)),
              en restant aligne sur le multiple choisi.

    force_target_downscale (bool) :
      False - comportement par defaut : si bbox > target, fallback cap MAX_SIDE.
      True  - si bbox > target, downscale GCD vers le target (meme algo que l'upscale).
              Ignore si target="none".

    Mode none (target="none") :
      Arrondit width/height au multiple choisi.
      Si le crop depasse MAX_SIDE (2048) : downscale GCD vers MAX_SIDE.

    Mode target (512/768/1024/1536/2048) :
      Upscale GCD vers target si bbox < target.
      Si force_target_downscale=True et bbox > target : downscale GCD vers target.
      Si force_target_downscale=False et bbox > target : cap MAX_SIDE (comportement historique).

    Anti-clamp : dans tous les cas, k est plafonne a l'espace disponible
      autour du centre du bbox -> le crop ne depasse jamais l'image source
      -> pas de clamp post-GCD -> ratio pixel-perfect garanti (0% de drift).

    Sorties :
      image_cropped / mask_cropped  -> VAE Encode (Inpaint)
      x / y                         -> ImageCompositeMasked
      orig_width / orig_height      -> dimensions crop DANS la source
      width / height                -> dimensions finales (apres resize)
      target_size                   -> valeur INT du target (0 si "none")
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":                  ("IMAGE",),
                "mask":                   ("MASK",),
                "x":                      ("INT", {"default": 0,  "min": 0, "max": 99999}),
                "y":                      ("INT", {"default": 0,  "min": 0, "max": 99999}),
                "width":                  ("INT", {"default": 64, "min": 1, "max": 99999}),
                "height":                 ("INT", {"default": 64, "min": 1, "max": 99999}),
                "multiple":               (["8 (VAE minimum)", "16 (Flux)", "32 (SD1.5)", "64 (SDXL)"],),
                "target":                 (TARGET_SIZES, {"default": "none"}),
                "force_square":           ("BOOLEAN", {"default": False}),
                "force_target_downscale": ("BOOLEAN", {"default": False}),
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

    def fix(self, image, mask, x, y, width, height, multiple, target,
            force_square=False, force_target_downscale=False):

        mult = int(multiple.split(" ")[0])
        B, H_src, W_src, C = image.shape

        # ── Step 1 : ratio force 1:1 si active ───────────────────────────────
        if force_square:
            base = max(width, height)
            base_w, base_h = base, base
        else:
            base_w, base_h = width, height

        # ── Step 2 : espace disponible autour du centre du bbox ───────────────
        # Garantit que le crop ne depasse jamais l'image -> pas de clamp post-GCD
        # -> ratio pixel-perfect dans tous les cas, y compris bords d'image.
        cx = x + width  // 2
        cy = y + height // 2
        avail_w = min(cx, W_src - cx) * 2
        avail_h = min(cy, H_src - cy) * 2
        # Plancher de securite : au minimum le bbox lui-meme
        avail_w = max(avail_w, base_w)
        avail_h = max(avail_h, base_h)

        # ── Step 3 : calcul new_w / new_h / up_w / up_h ──────────────────────
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

            if t_w > base_w and t_h > base_h:
                # Upscale vers target
                g    = math.gcd(t_w, t_h)
                a, b = t_w // g, t_h // g
                k    = round((base_w / a + base_h / b) / 2)
                k    = max(1, min(k, avail_w // a, avail_h // b))  # anti-clamp
                new_w, new_h = a * k, b * k
                up_w, up_h   = t_w, t_h
                need_resize  = True
                resize_label = "upscale "

            elif force_target_downscale:
                # Downscale GCD vers target
                g    = math.gcd(t_w, t_h)
                a, b = t_w // g, t_h // g
                k    = round((base_w / a + base_h / b) / 2)
                k    = max(1, min(k, avail_w // a, avail_h // b))  # anti-clamp
                new_w, new_h = a * k, b * k
                up_w, up_h   = t_w, t_h
                need_resize  = True
                resize_label = "downscale"

            else:
                # Fallback : arrondi au multiple + cap MAX_SIDE
                print(f"[BBoxMultipleFix] target={t} <= bbox -> fallback cap {MAX_SIDE}px")
                new_w = math.ceil(base_w / mult) * mult
                new_h = math.ceil(base_h / mult) * mult

                if new_w > MAX_SIDE or new_h > MAX_SIDE:
                    if new_w >= new_h:
                        down_w = MAX_SIDE
                        down_h = math.ceil((new_h * MAX_SIDE / new_w) / mult) * mult
                    else:
                        down_h = MAX_SIDE
                        down_w = math.ceil((new_w * MAX_SIDE / new_h) / mult) * mult

                    g    = math.gcd(down_w, down_h)
                    a, b = down_w // g, down_h // g
                    k    = round((new_w / a + new_h / b) / 2)
                    k    = max(1, min(k, avail_w // a, avail_h // b))  # anti-clamp
                    new_w, new_h = a * k, b * k
                    up_w, up_h   = down_w, down_h
                    need_resize  = True
                    resize_label = "downscale"
                else:
                    up_w, up_h = new_w, new_h

        else:
            # Mode none : arrondi au multiple, cap MAX_SIDE si depassement
            new_w = math.ceil(base_w / mult) * mult
            new_h = math.ceil(base_h / mult) * mult

            if new_w > MAX_SIDE or new_h > MAX_SIDE:
                if new_w >= new_h:
                    down_w = MAX_SIDE
                    down_h = math.ceil((new_h * MAX_SIDE / new_w) / mult) * mult
                else:
                    down_h = MAX_SIDE
                    down_w = math.ceil((new_w * MAX_SIDE / new_h) / mult) * mult

                g    = math.gcd(down_w, down_h)
                a, b = down_w // g, down_h // g
                k    = round((new_w / a + new_h / b) / 2)
                k    = max(1, min(k, avail_w // a, avail_h // b))  # anti-clamp
                new_w, new_h = a * k, b * k
                up_w, up_h   = down_w, down_h
                need_resize  = True
                resize_label = "downscale"
            else:
                up_w, up_h = new_w, new_h

        # ── Step 4 : expansion symetrique autour du bbox original ─────────────
        # Grace a l'anti-clamp, new_w/new_h rentrent toujours dans l'image —
        # le clamp ci-dessous ne modifie donc rien dans les cas normaux.
        new_x = x - (new_w - width)  // 2
        new_y = y - (new_h - height) // 2

        new_x = max(0, new_x)
        new_y = max(0, new_y)
        if new_x + new_w > W_src: new_x = W_src - new_w
        if new_y + new_h > H_src: new_y = H_src - new_h
        new_w = min(new_w, W_src)
        new_h = min(new_h, H_src)
        new_x = max(0, new_x)
        new_y = max(0, new_y)

        orig_w, orig_h = new_w, new_h

        flags = []
        if force_square:           flags.append("square")
        if force_target_downscale: flags.append("target_downscale")
        flag_info = f"  [{', '.join(flags)}]" if flags else ""
        print(f"[BBoxMultipleFix] bbox     : {width}x{height} @({x},{y}){flag_info}")
        print(f"[BBoxMultipleFix] crop     : {orig_w}x{orig_h} @({new_x},{new_y})")

        # ── Step 5 : crop ─────────────────────────────────────────────────────
        img_cropped = image[:, new_y:new_y + new_h, new_x:new_x + new_w, :]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        mask_cropped = mask[:, new_y:new_y + new_h, new_x:new_x + new_w]

        # ── Step 6 : resize Lanczos si necessaire ─────────────────────────────
        if need_resize and (up_w != orig_w or up_h != orig_h):
            print(f"[BBoxMultipleFix] {resize_label} : {orig_w}x{orig_h} -> {up_w}x{up_h}")
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
