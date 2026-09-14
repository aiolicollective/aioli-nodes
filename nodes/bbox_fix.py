"""
LEGACY — conserve pour les workflows existants.

BBoxMultipleFix s'inserait apres 'Mask Bounding Box' (ComfyUI Essentials) et
recevait son x / y / width / height. Les deux nodes sont remplaces par un seul,
[AioliMaskBBox](mask_bbox.py) : meme algorithme, mais il detecte le bbox
lui-meme a partir du mask.

Ce fichier reste en place pour que les anciens workflows continuent de se
charger a l'identique : cle "BBoxMultipleFix", memes entrees, memes sorties dans
le meme ordre. Le calcul vient desormais de bbox_core, partage avec le node
courant — il n'y a plus qu'une seule implementation de l'algorithme.

Ne rien ajouter ici : les evolutions vont dans bbox_core.py et mask_bbox.py.
"""

from .bbox_core import MAX_SIDE, MULTIPLES, TARGET_SIZES, compute_crop  # noqa: F401
from .tensor_utils import resize_lanczos


LOG = "BBoxMultipleFix"


class BBoxMultipleFix:
    """Voir AioliMaskBBox — ce node est son predecesseur, conserve tel quel."""

    DEPRECATED = True

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
    FUNCTION = "fix"
    CATEGORY = "Aioli Nodes"

    def fix(self, image, mask, x, y, width, height, multiple, target,
            force_square=False, force_target_downscale=False):

        B, H_src, W_src, C = image.shape

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

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        ny, nx = c["y"], c["x"]
        nh, nw = c["orig_height"], c["orig_width"]
        img_cropped = image[:, ny:ny + nh, nx:nx + nw, :]
        mask_cropped = mask[:, ny:ny + nh, nx:nx + nw]

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
