import torch

from .bbox_core import MULTIPLES, TARGET_SIZES, compute_crop
from .tensor_utils import resize_lanczos


LOG = "AioliMaskBBox"


class AioliMaskBBox:
    """
    Un seul node la ou il en fallait deux : 'Mask Bounding Box' (ComfyUI
    Essentials, en maintenance seule depuis avril 2025) suivi de
    'BBox Multiple Fix'.

    Prend l'image source + le mask, en deduit le bbox, l'arrondit au multiple
    choisi, gere la mise a l'echelle (up ou down) vers une resolution
    Flux-friendly, et recadre image + mask.

    padding (INT) :
      Marge en pixels ajoutee autour du bbox detecte, avant tout arrondi.
      Remplace a la fois le 'padding' et le 'blur' de Mask Bounding Box :
      les deux ne servaient qu'a elargir le bbox, un seul reglage suffit.

    force_square (bool) :
      False - ratio libre du bbox.
      True  - crop carre (cote = max(width, height)), aligne sur le multiple.
              NOTE : si max(width, height) > min(image_w, image_h), le carre
              theorique ne tient pas dans la source. Le crop final est alors
              rectangulaire mais la cible de resize reste carree -> l'image est
              etiree. Pour un recompose pixel-perfect dans ce cas, connecter un
              ImageResize+ en mode 'stretch' (keep_proportion=False) en sortie
              du modele, avec orig_width / orig_height comme cibles. Un
              avertissement est logue quand le cas se produit.

    force_target_downscale (bool) :
      False - si bbox > target, fallback cap 2048 px.
      True  - si bbox > target, downscale GCD vers le target (meme algo que
              l'upscale). Ignore si target="none".

    Anti-clamp : le crop est plafonne a l'espace disponible autour du centre du
    bbox, donc il ne deborde jamais de l'image source -> ratio pixel-perfect
    garanti (0% de drift), y compris quand la zone masquee touche un bord.

    Sorties :
      image_cropped / mask_cropped  -> VAE Encode (Inpaint)
      x / y                         -> ImageCompositeMasked
      orig_width / orig_height      -> dimensions du crop DANS la source
      width / height                -> dimensions finales (apres resize)
      target_size                   -> valeur INT du target (0 si "none")
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":                  ("IMAGE",),
                "mask":                   ("MASK",),
                "padding":                ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "multiple":               (MULTIPLES,),
                "target":                 (TARGET_SIZES, {"default": "none"}),
                "force_square":           ("BOOLEAN", {"default": False}),
                "force_target_downscale": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("image_cropped", "mask_cropped",
                    "x", "y",
                    "orig_width", "orig_height",
                    "width", "height",
                    "target_size")
    FUNCTION = "execute"
    CATEGORY = "Aioli Nodes"

    def execute(self, image, mask, padding, multiple, target,
                force_square=False, force_target_downscale=False):

        B, H_src, W_src, C = image.shape

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        # Le crop se fait en coordonnees de l'image source : si le mask n'a pas
        # la meme definition, c'est lui qu'on aligne (l'inverse de Mask Bounding
        # Box, qui redimensionnait l'image).
        if mask.shape[1] != H_src or mask.shape[2] != W_src:
            print(f"[{LOG}] mask {mask.shape[2]}x{mask.shape[1]} != image {W_src}x{H_src} "
                  f"-> mask redimensionne sur l'image")
            mask = resize_lanczos(mask, W_src, H_src, "mask")

        # -- bbox du mask ---------------------------------------------------
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

        # -- calcul du crop -------------------------------------------------
        c = compute_crop(x, y, width, height, W_src, H_src, multiple, target,
                         force_square, force_target_downscale, log_prefix=LOG)

        flags = []
        if force_square:
            flags.append("square")
        if force_target_downscale:
            flags.append("target_downscale")
        flag_info = f"  [{', '.join(flags)}]" if flags else ""
        print(f"[{LOG}] bbox     : {width}x{height} @({x},{y}){flag_info}")
        print(f"[{LOG}] crop     : {c['orig_width']}x{c['orig_height']} @({c['x']},{c['y']})")

        # -- crop ------------------------------------------------------------
        ny, nx = c["y"], c["x"]
        nh, nw = c["orig_height"], c["orig_width"]
        img_cropped = image[:, ny:ny + nh, nx:nx + nw, :]
        mask_cropped = mask[:, ny:ny + nh, nx:nx + nw]

        # -- resize Lanczos si necessaire --------------------------------------
        if c["need_resize"]:
            up_w, up_h = c["resize_to"]
            print(f"[{LOG}] {c['resize_label']} : {nw}x{nh} -> {up_w}x{up_h}")
            img_cropped = resize_lanczos(img_cropped, up_w, up_h, "image")
            mask_cropped = resize_lanczos(mask_cropped, up_w, up_h, "mask")

        return (img_cropped, mask_cropped,
                c["x"], c["y"],
                c["orig_width"], c["orig_height"],
                c["width"], c["height"],
                c["target_size"])
