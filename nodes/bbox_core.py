"""
Coeur de calcul partage par les nodes bbox d'aioli-nodes.

Fonctions pures : a partir d'un bbox (x, y, width, height) et des dimensions de
l'image source, calcule le crop final et la cible de resize. Aucune dependance
torch / PIL ici — uniquement des entiers, ce qui rend l'algorithme testable hors
de ComfyUI.

Utilise par :
  - AioliMaskBBox   (nodes/mask_bbox.py) — node courant
  - BBoxMultipleFix (nodes/bbox_fix.py)  — node legacy, conserve pour les
                                           anciens workflows

Toute evolution de l'algorithme se fait ICI, jamais dans les nodes.
"""

import math


TARGET_SIZES = ["none", "512", "768", "1024", "1536", "2048"]
MULTIPLES = ["8 (VAE minimum)", "16 (Flux)", "32 (SD1.5)", "64 (SDXL)"]

# Plafond dur pour compatibilite Flux / SDXL — utilise en mode none, et en mode
# target quand force_target_downscale=False.
MAX_SIDE = 2048


def _gcd_fit(t_w, t_h, ref_w, ref_h, avail_w, avail_h):
    """
    Ramene (t_w, t_h) a sa forme irreductible (a, b), puis cherche le multiplieur
    k le plus proche de la taille de reference — plafonne a l'espace disponible.
    Garantit un ratio strictement egal a celui de la cible (0% de drift).
    """
    g = math.gcd(t_w, t_h)
    a, b = t_w // g, t_h // g
    k = round((ref_w / a + ref_h / b) / 2)
    k = max(1, min(k, avail_w // a, avail_h // b))  # anti-clamp
    return a * k, b * k


def _cap_to_max_side(w, h, mult):
    """Reduit (w, h) pour que le plus grand cote tienne dans MAX_SIDE."""
    if w >= h:
        down_w = MAX_SIDE
        down_h = math.ceil((h * MAX_SIDE / w) / mult) * mult
    else:
        down_h = MAX_SIDE
        down_w = math.ceil((w * MAX_SIDE / h) / mult) * mult
    return down_w, down_h


def compute_crop(x, y, width, height, src_w, src_h, multiple, target,
                 force_square=False, force_target_downscale=False,
                 log_prefix="AioliMaskBBox"):
    """
    Entrees  : bbox (x, y, width, height) en coordonnees de l'image source,
               dimensions source, et les reglages du node.
    Sortie   : dict decrivant le crop a effectuer et la cible de resize.

      x, y                     -> origine du crop dans la source
      orig_width, orig_height  -> dimensions du crop DANS la source
      width, height            -> dimensions finales (apres resize eventuel)
      target_size              -> valeur INT du target (0 si "none")
      need_resize              -> bool
      resize_to                -> (w, h) cible du resize
      resize_label             -> "upscale " / "downscale" / ""

    Anti-clamp : le crop est toujours contraint a l'espace disponible autour du
    centre du bbox, donc il ne deborde jamais de l'image source. Pas de clamp
    apres le calcul GCD -> ratio pixel-perfect garanti, y compris en bord d'image.
    """
    mult = int(str(multiple).split(" ")[0])

    # -- Step 1 : ratio force 1:1 si active -------------------------------
    if force_square:
        base = max(width, height)
        base_w, base_h = base, base
        # Le carre theorique ne tient pas toujours dans la source : le clamp
        # final reduit alors le crop a un rectangle, mais la cible de resize
        # reste carree -> l'image est etiree. Pour un recompose pixel-perfect
        # dans ce cas, connecter un ImageResize+ en mode 'stretch'
        # (keep_proportion=False) en sortie du modele, avec orig_width /
        # orig_height comme cibles.
        if base > min(src_w, src_h):
            print(f"[{log_prefix}] WARNING : force_square + bbox larger than smallest source dim "
                  f"({base}px > min({src_w},{src_h})). Crop will be non-square and stretched to fit "
                  f"a square. Set downstream ImageResize+ to 'stretch' mode (keep_proportion=False) "
                  f"using orig_width/orig_height for pixel-perfect recompose.")
    else:
        base_w, base_h = width, height

    # -- Step 2 : espace disponible autour du centre du bbox ---------------
    cx = x + width // 2
    cy = y + height // 2
    avail_w = min(cx, src_w - cx) * 2
    avail_h = min(cy, src_h - cy) * 2
    # Plancher de securite : au minimum le bbox lui-meme
    avail_w = max(avail_w, base_w)
    avail_h = max(avail_h, base_h)

    # -- Step 3 : calcul new_w / new_h / up_w / up_h -----------------------
    need_resize = False
    resize_label = ""
    up_w = up_h = 0

    if target != "none":
        t = int(target)

        # Dimensions cible en multiple de mult, au ratio du crop de base
        if base_w >= base_h:
            t_w = t
            t_h = math.ceil((base_h * t / base_w) / mult) * mult
        else:
            t_h = t
            t_w = math.ceil((base_w * t / base_h) / mult) * mult

        if t_w > base_w and t_h > base_h:
            # Upscale vers target
            new_w, new_h = _gcd_fit(t_w, t_h, base_w, base_h, avail_w, avail_h)
            up_w, up_h = t_w, t_h
            need_resize = True
            resize_label = "upscale "

        elif force_target_downscale:
            # Downscale GCD vers target
            new_w, new_h = _gcd_fit(t_w, t_h, base_w, base_h, avail_w, avail_h)
            up_w, up_h = t_w, t_h
            need_resize = True
            resize_label = "downscale"

        else:
            # Fallback : arrondi au multiple + cap MAX_SIDE
            print(f"[{log_prefix}] target={t} <= bbox -> fallback cap {MAX_SIDE}px")
            new_w = math.ceil(base_w / mult) * mult
            new_h = math.ceil(base_h / mult) * mult

            if new_w > MAX_SIDE or new_h > MAX_SIDE:
                down_w, down_h = _cap_to_max_side(new_w, new_h, mult)
                new_w, new_h = _gcd_fit(down_w, down_h, new_w, new_h, avail_w, avail_h)
                up_w, up_h = down_w, down_h
                need_resize = True
                resize_label = "downscale"
            else:
                up_w, up_h = new_w, new_h

    else:
        # Mode none : arrondi au multiple, cap MAX_SIDE si depassement
        new_w = math.ceil(base_w / mult) * mult
        new_h = math.ceil(base_h / mult) * mult

        if new_w > MAX_SIDE or new_h > MAX_SIDE:
            down_w, down_h = _cap_to_max_side(new_w, new_h, mult)
            new_w, new_h = _gcd_fit(down_w, down_h, new_w, new_h, avail_w, avail_h)
            up_w, up_h = down_w, down_h
            need_resize = True
            resize_label = "downscale"
        else:
            up_w, up_h = new_w, new_h

    # -- Step 4 : expansion symetrique autour du bbox original -------------
    # Grace a l'anti-clamp, new_w/new_h rentrent toujours dans l'image : le
    # clamp ci-dessous ne modifie rien dans les cas normaux.
    new_x = x - (new_w - width) // 2
    new_y = y - (new_h - height) // 2

    new_x = max(0, new_x)
    new_y = max(0, new_y)
    if new_x + new_w > src_w:
        new_x = src_w - new_w
    if new_y + new_h > src_h:
        new_y = src_h - new_h
    new_w = min(new_w, src_w)
    new_h = min(new_h, src_h)
    new_x = max(0, new_x)
    new_y = max(0, new_y)

    orig_w, orig_h = new_w, new_h

    if need_resize and (up_w != orig_w or up_h != orig_h):
        final_w, final_h = up_w, up_h
    else:
        need_resize = False
        final_w, final_h = orig_w, orig_h

    return {
        "x": new_x,
        "y": new_y,
        "orig_width": orig_w,
        "orig_height": orig_h,
        "width": final_w,
        "height": final_h,
        "target_size": 0 if target == "none" else int(target),
        "need_resize": need_resize,
        "resize_to": (up_w, up_h),
        "resize_label": resize_label,
    }
