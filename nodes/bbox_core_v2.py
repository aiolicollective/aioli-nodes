"""
Coeur de calcul du node Aioli Mask BBox V2 (cle "AioliMaskBBoxV2").

Fonctions pures : a partir d'un bbox (x, y, width, height) et des dimensions de
l'image source, decide la taille envoyee au modele et le crop a prelever dans
la source. Aucune dependance torch / PIL — uniquement des nombres, donc
testable hors de ComfyUI.

bbox_core.py n'est PAS touche : il reste le calcul des deux nodes legacy
(AioliMaskBBox v1.6, BBoxMultipleFix). Les evolutions du V2 se font ICI.

Regles (spec v1.7.0) :

  resolution N (INT, 0 = pas de cible)
    Taille modele en "equivalent carre" : autant de pixels que N x N, au ratio
    du crop, cotes au multiple. Carre : N x N exactement. Paire de multiples
    choisie au plus proche du ratio, a +/-5 % de N^2.

  Plafond en pixels : 2048^2, ou N^2 si N > 2048.
    crop < N^2                          -> agrandi a N^2
    crop > N^2, force_target_downscale  -> reduit a N^2
    crop > N^2, sinon                   -> taille native, plafonnee
    N = 0                               -> taille native, plafonnee

  Region tres allongee : la taille modele ne depasse pas 4:1 (ou 1:4). Une
  region plus fine (trait de 1:10) garde un crop au plus 4:1, elargi de
  contexte sur le petit cote : jamais de modele a 64 x 3200.

  Geometrie, regle unique : le crop est le bbox agrandi au ratio de la taille
  modele, centre sur le bbox, rentre dans l'image. S'il depasse l'image, des
  BANDES completent (x / y negatifs possibles, orig_width / orig_height =
  taille du crop bandes comprises). Jamais d'etirement au-dela de l'arrondi
  au pixel (< 0,5 px par cote), jamais de bbox coupe.
"""

import math


MULTIPLES = ["8 (VAE minimum)", "16 (Flux)", "32 (SD1.5)", "64 (SDXL)"]

CAP_SIDE = 2048            # plafond par defaut : 2048^2 pixels (4,2 MP)
WARN_RESOLUTION = 2048     # au-dessus : avertissement (VRAM, temps)
MAX_RESOLUTION = 4096
AREA_TOLERANCE = 0.05      # +/-5 % autour de la surface visee
MAX_ASPECT = 4.0           # taille modele jamais plus allongee que 4:1 / 1:4


def parse_multiple(multiple):
    """'16 (Flux)' -> 16. Accepte aussi un int."""
    return int(str(multiple).split(" ")[0])


def _round_half_up(v):
    return int(math.floor(v + 0.5))


def pick_model_size(ratio, area, mult):
    """
    Paire (w, h), multiples de `mult`, de ratio w/h le plus proche de `ratio`
    parmi celles dont la surface est a +/-5 % de `area`. A ratio egal, la
    surface la plus proche gagne. Aucune paire dans la tolerance (cas extremes)
    -> surface la plus proche, puis ratio.

    Carre (ratio 1) et N multiple de `mult` -> exactement N x N.
    Paires plus allongees que 4:1 exclues.

    Enumeration exhaustive par le petit cote s (s <= sqrt(1,05 x area)) : pour
    chaque s, le grand cote candidat est celui du ratio, celui de la surface,
    ou une borne de la tolerance — l'optimum en fait toujours partie.
    """
    if ratio <= 0 or area <= 0:
        raise ValueError(f"pick_model_size: ratio={ratio}, area={area}")

    lo_area = area * (1.0 - AREA_TOLERANCE)
    hi_area = area * (1.0 + AREA_TOLERANCE)
    k_short_max = max(1, int(math.sqrt(hi_area) // mult))

    best = None       # (cle, w, h) dans la tolerance
    fallback = None   # (cle, w, h) hors tolerance

    def consider(w, h):
        nonlocal best, fallback
        if max(w, h) > MAX_ASPECT * min(w, h) + 1e-9:
            return           # jamais plus allonge que 4:1 (le carre reste toujours candidat)
        r_err = round(abs(math.log((w / h) / ratio)), 12)
        a_err = round(abs(w * h / area - 1.0), 12)
        if lo_area - 1e-6 <= w * h <= hi_area + 1e-6:
            key = (r_err, a_err, w)
            if best is None or key < best[0]:
                best = (key, w, h)
        else:
            key = (a_err, r_err, w)
            if fallback is None or key < fallback[0]:
                fallback = (key, w, h)

    for ks in range(1, k_short_max + 1):
        s = ks * mult
        # s = largeur (petit cote) puis s = hauteur (petit cote)
        for long_ratio, as_width in ((s / ratio, True), (s * ratio, False)):
            kls = set()
            for v in (long_ratio, area / s):
                k = v / mult
                kls.add(math.floor(k))
                kls.add(math.ceil(k))
            kls.add(math.ceil(lo_area / s / mult))
            kls.add(math.floor(hi_area / s / mult))
            for kl in kls:
                if kl < ks:          # le grand cote reste >= au petit
                    continue
                L = kl * mult
                if as_width:
                    consider(s, L)
                else:
                    consider(L, s)

    chosen = best if best is not None else fallback
    return chosen[1], chosen[2]


def _place(b0, b_len, c_len, src_len):
    """
    Origine du crop sur un axe : centre sur le bbox, puis rentre dans l'image.
    Si le crop est plus grand que l'image, il la couvre entierement (origine
    negative, bandes) en restant au plus pres du centre du bbox.
    """
    pos = b0 - (c_len - b_len) // 2
    lo, hi = min(0, src_len - c_len), max(0, src_len - c_len)
    return min(max(pos, lo), hi)


def compute_crop_v2(x, y, width, height, src_w, src_h, multiple, resolution,
                    force_square=False, force_target_downscale=False,
                    log_prefix="AioliMaskBBoxV2"):
    """
    Entrees : bbox (x, y, width, height) en coordonnees source (dans l'image),
              dimensions source, reglages du node.

    Sortie (dict) :
      x, y                     origine du crop dans la source (peut etre < 0)
      orig_width, orig_height  taille du crop dans la source, bandes comprises
      width, height            taille envoyee au modele
      target_size              resolution N (0 = pas de cible)
      need_resize              bool
      resize_label             "upscale" / "downscale" / ""
      scale                    pixels source par pixel modele
      bands                    (gauche, haut, droite, bas) en pixels source
      image_rect               (x0, y0, x1, y1) de l'image dans le crop
    """
    mult = parse_multiple(multiple)
    N = max(0, int(resolution))
    bw, bh = int(width), int(height)
    if bw <= 0 or bh <= 0:
        raise ValueError(f"[{log_prefix}] bbox vide : {bw}x{bh}")

    if N > WARN_RESOLUTION:
        print(f"[{log_prefix}] WARNING : resolution {N} > {WARN_RESOLUTION} "
              f"({N * N / 1e6:.1f} MP au modele) -> VRAM et temps en hausse.")

    cap_area = max(CAP_SIDE * CAP_SIDE, N * N)

    # Region tres allongee : le petit cote est elargi (contexte) jusqu'a 1:4.
    ew = max(bw, bh / MAX_ASPECT)
    eh = max(bh, bw / MAX_ASPECT)
    ratio = 1.0 if force_square else ew / eh

    # Taille native : le crop arrondi au multiple superieur, a l'echelle 1.
    if force_square:
        s = math.ceil(max(bw, bh) / mult) * mult
        native = (s, s)
    else:
        native = (math.ceil(ew / mult) * mult, math.ceil(eh / mult) * mult)

    def fit_scale(model):
        """Plus petite echelle pour qu'un crop au ratio du modele contienne le bbox."""
        return max(bw / model[0], bh / model[1])

    def native_capped():
        if native[0] * native[1] <= cap_area:
            return native, 1.0
        capped = pick_model_size(ratio, cap_area, mult)
        if capped[0] * capped[1] >= native[0] * native[1]:
            return native, 1.0
        return capped, fit_scale(capped)

    if N > 0:
        target = pick_model_size(ratio, N * N, mult)
        s_t = fit_scale(target)
        if s_t < 1.0 or force_target_downscale:
            model, scale = target, s_t
        else:
            model, scale = native_capped()
    else:
        model, scale = native_capped()

    # Taille du crop dans la source, au ratio du modele.
    if scale == 1.0:
        cw, ch = model
    else:
        cw = max(bw, _round_half_up(model[0] * scale))
        ch = max(bh, _round_half_up(model[1] * scale))

    cx = _place(int(x), bw, cw, src_w)
    cy = _place(int(y), bh, ch, src_h)

    band_l = max(0, -cx)
    band_t = max(0, -cy)
    band_r = max(0, cx + cw - src_w)
    band_b = max(0, cy + ch - src_h)

    need_resize = (cw, ch) != tuple(model)
    if not need_resize:
        label = ""
    elif model[0] * model[1] > cw * ch:
        label = "upscale"
    else:
        label = "downscale"

    return {
        "x": cx,
        "y": cy,
        "orig_width": cw,
        "orig_height": ch,
        "width": model[0],
        "height": model[1],
        "target_size": N,
        "need_resize": need_resize,
        "resize_label": label,
        "scale": scale,
        "bands": (band_l, band_t, band_r, band_b),
        "image_rect": (band_l, band_t, cw - band_r, ch - band_b),
    }
