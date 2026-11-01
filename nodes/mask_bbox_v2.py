import math

import torch

from .bbox_core_v2 import (MAX_RESOLUTION, MULTIPLES,
                           compute_crop_v2)
from .tensor_utils import resize_lanczos


LOG = "AioliMaskBBoxV2"

CONTEXTS = ["region", "full_image"]


def _crop_replicate(t, x, y, cw, ch):
    """
    Preleve le rectangle (x, y, cw, ch) dans t ; hors image, le bord de l'image
    est prolonge (replicate). Sert au resize : aucun gris ne se melange aux
    pixels de bord. t : (B, H, W, C) ou (B, H, W).
    """
    H, W = t.shape[1], t.shape[2]
    rows = (torch.arange(ch) + y).clamp(0, H - 1)
    cols = (torch.arange(cw) + x).clamp(0, W - 1)
    return t[:, rows][:, :, cols]


def _paint_bands(t, x0, y0, x1, y1, fill):
    """Peint `fill` hors de [x0, x1) x [y0, y1) (bandes, a la taille modele)."""
    t = t.clone()
    t[:, :y0] = fill
    t[:, y1:] = fill
    t[:, :, :x0] = fill
    t[:, :, x1:] = fill
    if t.dim() == 4 and t.shape[3] == 4:
        a = t[..., 3]
        a[:, :y0] = 1.0
        a[:, y1:] = 1.0
        a[:, :, :x0] = 1.0
        a[:, :, x1:] = 1.0
    return t


def _crop_with_bands(t, x, y, cw, ch, fill):
    """
    Preleve le rectangle (x, y, cw, ch) dans t, qui peut deborder de l'image.
    t : (B, H, W, C) ou (B, H, W). Hors image -> `fill` (les bandes).
    Sans debordement : simple slicing, pixels identiques a la source.
    """
    H, W = t.shape[1], t.shape[2]
    if x >= 0 and y >= 0 and x + cw <= W and y + ch <= H:
        return t[:, y:y + ch, x:x + cw]
    out = torch.full((t.shape[0], ch, cw) + tuple(t.shape[3:]), float(fill),
                     dtype=t.dtype, device=t.device)
    if t.dim() == 4 and t.shape[3] == 4:
        out[..., 3] = 1.0                       # RGBA : bandes opaques
    sx0, sy0 = max(0, x), max(0, y)
    sx1, sy1 = min(W, x + cw), min(H, y + ch)
    if sx1 > sx0 and sy1 > sy0:
        out[:, sy0 - y:sy1 - y, sx0 - x:sx1 - x] = t[:, sy0:sy1, sx0:sx1]
    return out


class AioliMaskBBoxV2:
    """
    Image + mask -> crop pret pour le modele, une seule regle de geometrie.
    Remplace 📐 Aioli Mask BBox (v1.6, garde en legacy). A recomposer avec
    🧩 BBox Multiple Assembler (x / y peuvent etre negatifs : ImageCompositeMasked
    ne convient plus).

    padding (INT)
      Marge en pixels autour du bbox detecte. Ignore en context = full_image.

    multiple
      Les cotes envoyes au modele sont multiples de cette valeur.

    resolution (INT, 0 = pas de cible)
      Taille modele en EQUIVALENT CARRE : autant de pixels que N x N, au ratio
      du crop. N = 2048 -> 1:1 2048x2048, 3:2 2496x1664, 16:9 2704x1520 (x16).
      Carre : N x N exactement. Au-dessus de 2048 : avertissement.
      Crop plus petit que N^2 -> agrandi a N^2.
      Crop plus grand que N^2 -> force_target_downscale on : reduit a N^2 ;
                                 off : taille native, plafonnee.
      N = 0 -> taille native (arrondie au multiple), plafonnee.
      Plafond : 2048^2 pixels (4,2 MP), ou N^2 si N > 2048.

    force_square (bool)
      Taille modele carree. Si le carre ne tient pas dans l'image, des bandes
      completent (jamais d'etirement).

    context
      region     : crop = le bbox du masque (+ padding), agrandi au ratio.
      full_image : crop = l'image entiere (+ bandes au ratio). Le masque reste
                   celui de la region : le modele voit tout, l'assembleur ne
                   recolle que la zone masquee.

    band_color (FLOAT, opt.)
      Gris des bandes (0,5 par defaut).

    Geometrie : le crop est le bbox agrandi au ratio de la taille modele, centre,
    rentre dans l'image. S'il deborde de l'image, des BANDES completent : x / y
    negatifs, orig_width / orig_height = taille du crop bandes comprises. Le bbox
    n'est jamais coupe.

    v1.8 : crop avec bandes ET resize -> le resize se fait avec le bord de
    l'image prolonge, puis les bandes sont peintes nettes a la taille modele
    (plus de gris melange aux pixels de bord). Sans bande ou sans resize :
    inchange.

    Masque vide : bbox = image entiere, masque a 0 (avertissement). Pour editer
    toute l'image, utiliser Mask Split Regions en empty_mask = full_image.

    Sorties : les 9 de Aioli Mask BBox, meme ordre, + image_area.
      image_cropped / mask_cropped  -> modele
      x / y                         -> BBox Multiple Assembler (peuvent etre < 0)
      orig_width / orig_height      -> BBox Multiple Assembler (width / height)
      width / height                -> taille envoyee au modele
      target_size                   -> resolution N (0 = pas de cible)
      image_area                    -> MASK a la taille modele : 1 dans l'image,
                                       0 dans les bandes et la frange de resize
                                       qui les borde (Inpaint Color Fix)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":                  ("IMAGE",),
                "mask":                   ("MASK",),
                "padding":                ("INT", {"default": 48, "min": 0, "max": 4096, "step": 1}),
                "multiple":               (MULTIPLES, {"default": "16 (Flux)"}),
                "resolution":             ("INT", {"default": 1024, "min": 0, "max": MAX_RESOLUTION,
                                                   "step": 64, "display": "slider",
                                                   "tooltip": "Taille modele en equivalent carre (N x N pixels). "
                                                              "0 = taille native. Un crop plus grand que N^2 n'est "
                                                              "reduit a N^2 que si force_target_downscale est on ; "
                                                              "sinon il garde sa taille native, plafonnee a 2048^2."}),
                "force_square":           ("BOOLEAN", {"default": False}),
                "force_target_downscale": ("BOOLEAN", {"default": False,
                                                       "tooltip": "On : un crop plus grand que N^2 est reduit a N^2. "
                                                                  "Off : taille native plafonnee (2048^2) — pour "
                                                                  "l'image entiere, resolution n'a alors plus d'effet."}),
                "context":                (CONTEXTS, {"default": "region"}),
            },
            "optional": {
                "band_color":             ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT", "INT", "INT", "INT", "MASK")
    RETURN_NAMES = ("image_cropped", "mask_cropped",
                    "x", "y",
                    "orig_width", "orig_height",
                    "width", "height",
                    "target_size",
                    "image_area")
    FUNCTION = "execute"
    CATEGORY = "Aioli Nodes"

    def execute(self, image, mask, padding, multiple, resolution,
                force_square=False, force_target_downscale=False,
                context="region", band_color=0.5):

        B, H_src, W_src, C = image.shape

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        # Crop en coordonnees image : c'est le mask qu'on aligne, pas l'image.
        if mask.shape[1] != H_src or mask.shape[2] != W_src:
            print(f"[{LOG}] mask {mask.shape[2]}x{mask.shape[1]} != image {W_src}x{H_src} "
                  f"-> mask redimensionne sur l'image")
            mask = resize_lanczos(mask, W_src, H_src, "mask")

        # -- bbox -------------------------------------------------------------
        if context == "full_image":
            x, y, width, height = 0, 0, W_src, H_src
        else:
            nz = torch.nonzero(mask > 0)
            if nz.numel() == 0:
                print(f"[{LOG}] WARNING : mask vide -> bbox = image entiere")
                x, y, width, height = 0, 0, W_src, H_src
            else:
                y_idx, x_idx = nz[:, 1], nz[:, 2]
                x1 = max(0, int(x_idx.min()) - padding)
                x2 = min(W_src, int(x_idx.max()) + 1 + padding)
                y1 = max(0, int(y_idx.min()) - padding)
                y2 = min(H_src, int(y_idx.max()) + 1 + padding)
                x, y, width, height = x1, y1, x2 - x1, y2 - y1

        # -- geometrie ----------------------------------------------------------
        c = compute_crop_v2(x, y, width, height, W_src, H_src, multiple, resolution,
                            force_square, force_target_downscale, log_prefix=LOG)

        flags = [context]
        if force_square:
            flags.append("square")
        if force_target_downscale:
            flags.append("target_downscale")
        print(f"[{LOG}] bbox     : {width}x{height} @({x},{y})  [{', '.join(flags)}]")
        print(f"[{LOG}] crop     : {c['orig_width']}x{c['orig_height']} @({c['x']},{c['y']})")
        if any(c["bands"]):
            l, t, r, b = c["bands"]
            print(f"[{LOG}] bandes   : gauche {l}, haut {t}, droite {r}, bas {b} (px source)")

        # -- crop (bandes comprises) + resize Lanczos vers la taille modele ---------
        cx, cy = c["x"], c["y"]
        cw, ch = c["orig_width"], c["orig_height"]
        mw, mh = c["width"], c["height"]
        ix0, iy0, ix1, iy1 = c["image_rect"]
        sx, sy = mw / cw, mh / ch
        # rectangle de l'image a la taille modele (bandes autour)
        ax0, ax1 = int(round(ix0 * sx)), int(round(ix1 * sx))
        ay0, ay1 = int(round(iy0 * sy)), int(round(iy1 * sy))
        bands = any(c["bands"])
        if c["need_resize"]:
            print(f"[{LOG}] {c['resize_label']} : {cw}x{ch} -> {mw}x{mh}")
            if bands:
                # v1.8 : resize avec le bord de l'image prolonge, puis bandes
                # peintes nettes a la taille modele. Avant, le gris des bandes
                # entrait dans le Lanczos et debordait dans l'image (liseres),
                # et le masque s'adoucissait au bord de l'image.
                img_cropped = resize_lanczos(_crop_replicate(image, cx, cy, cw, ch), mw, mh, "image")
                mask_cropped = resize_lanczos(_crop_replicate(mask, cx, cy, cw, ch), mw, mh, "mask")
                img_cropped = _paint_bands(img_cropped, ax0, ay0, ax1, ay1, band_color)
                mask_cropped = _paint_bands(mask_cropped, ax0, ay0, ax1, ay1, 0.0)
            else:
                img_cropped = resize_lanczos(_crop_with_bands(image, cx, cy, cw, ch, band_color), mw, mh, "image")
                mask_cropped = resize_lanczos(_crop_with_bands(mask, cx, cy, cw, ch, 0.0), mw, mh, "mask")
        else:
            print(f"[{LOG}] modele   : {mw}x{mh} (pas de resize)")
            img_cropped = _crop_with_bands(image, cx, cy, cw, ch, band_color)
            mask_cropped = _crop_with_bands(mask, cx, cy, cw, ch, 0.0)

        # -- image_area : 1 dans l'image, 0 dans les bandes (taille modele) -------
        # Cote bande, on retire une frange (rayon Lanczos 3 px, a l'echelle du
        # resize) : le modele voit la limite gris / image, zone de mesure propre.
        if c["need_resize"]:
            ex, ey = math.ceil(3 * max(1.0, sx)), math.ceil(3 * max(1.0, sy))
            bl, bt, br, bb = c["bands"]
            ax0 += ex if bl else 0
            ax1 -= ex if br else 0
            ay0 += ey if bt else 0
            ay1 -= ey if bb else 0
        image_area = torch.zeros((B, mh, mw), dtype=torch.float32)
        if ax1 > ax0 and ay1 > ay0:
            image_area[:, ay0:ay1, ax0:ax1] = 1.0

        return (img_cropped, mask_cropped,
                cx, cy,
                cw, ch,
                mw, mh,
                c["target_size"],
                image_area)
