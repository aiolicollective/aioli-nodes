import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from .tensor_utils import to_uint8

_PALETTE = [
    (1.0, 0.2, 0.2), (0.2, 0.8, 1.0), (0.4, 1.0, 0.3), (1.0, 0.85, 0.1),
    (0.9, 0.3, 1.0), (1.0, 0.55, 0.0), (0.2, 1.0, 0.8), (1.0, 0.4, 0.7),
]


def _resize_img(hwc, new_w, new_h):
    arr = to_uint8(hwc.cpu().numpy())
    pil = Image.fromarray(arr).resize((new_w, new_h), Image.LANCZOS)
    return torch.from_numpy(np.array(pil).astype(np.float32) / 255.0)


def _resize_mask(hw, new_w, new_h):
    arr = to_uint8(hw.cpu().numpy())
    pil = Image.fromarray(arr, mode='L').resize((new_w, new_h), Image.LANCZOS)
    return torch.from_numpy(np.array(pil).astype(np.float32) / 255.0)


def _replicate_outside(t, x0, y0, x1, y1):
    """
    Remplace tout ce qui est hors du rectangle [x0, x1) x [y0, y1) par le bord
    du rectangle (replicate). t : (H, W) ou (H, W, C).
    """
    H, W = t.shape[0], t.shape[1]
    if x0 <= 0 and y0 <= 0 and x1 >= W and y1 >= H:
        return t
    rows = torch.arange(H).clamp(y0, y1 - 1)
    cols = torch.arange(W).clamp(x0, x1 - 1)
    return t[rows][:, cols]


def _sep_maxpool(m, px):
    """Dilatation (grow) separable -> O(H*W*k) au lieu de O(H*W*k^2)."""
    k = int(px) * 2 + 1
    m = F.max_pool2d(m, (1, k), stride=1, padding=(0, int(px)))
    m = F.max_pool2d(m, (k, 1), stride=1, padding=(int(px), 0))
    return m


def _grow_shrink_fast(mask_hw, px):
    """
    Grow (px>0) / shrink (px<0) sur la PLEINE TOILE, borne par l'image.
    Rapide meme pour de grandes valeurs : au-dela de 24 px on travaille sur un
    masque sous-echantillonne (le rayon effectif reste petit), puis on re-upscale.
    """
    if px == 0:
        return mask_hw
    px = int(px)
    a = abs(px)
    m = mask_hw.unsqueeze(0).unsqueeze(0).float()

    # downscale pour les gros rayons : kernel effectif plafonne ~24
    scale = 1
    if a > 24:
        scale = int(np.ceil(a / 24.0))
        H, W = m.shape[-2:]
        m = F.avg_pool2d(m, scale, stride=scale, ceil_mode=True)
    r = max(1, round(a / scale))

    if px > 0:
        m = _sep_maxpool(m, r)
    else:
        m = 1.0 - _sep_maxpool(1.0 - m, r)

    if scale > 1:
        m = F.interpolate(m, size=mask_hw.shape, mode="bilinear", align_corners=False)
    return m.squeeze(0).squeeze(0).clamp(0.0, 1.0)


def _feather_fast(mask_hw, px):
    """
    Flou box separable (feather) sur la pleine toile.

    v1.7.0 : bord de l'image en padding 'replicate'. Avant, avg_pool2d comptait
    le bord comme du noir : un masque qui touchait le bord de l'image y tombait
    a ~0,53 (~0,28 en coin, feather 8) et l'original revenait. Avec 'replicate',
    le bord prolonge le masque tel qu'il est : un masque qui touche le bord
    reste a 1, un masque qui s'arrete avant le bord n'est pas sur-corrige.
    """
    if px <= 0:
        return mask_hw
    px = int(px)
    m = mask_hw.unsqueeze(0).unsqueeze(0).float()
    scale = 1
    if px > 24:
        scale = int(np.ceil(px / 24.0))
        m = F.avg_pool2d(m, scale, stride=scale, ceil_mode=True)
    r = max(1, round(px / scale))
    k = r * 2 + 1
    m = F.avg_pool2d(F.pad(m, (r, r, 0, 0), mode="replicate"), (1, k), stride=1)
    m = F.avg_pool2d(F.pad(m, (0, 0, r, r), mode="replicate"), (k, 1), stride=1)
    if scale > 1:
        m = F.interpolate(m, size=mask_hw.shape, mode="bilinear", align_corners=False)
    return m.squeeze(0).squeeze(0).clamp(0.0, 1.0)


class BBoxMultipleAssembler:
    """
    Recompose pixel-perfect d'une LISTE de crops sur une seule image de base.

    Chaque masque (SAM3 / multi-bbox) est recadre puis enhance separement dans le
    KSampler ; ce node replie la liste des N crops a leurs coordonnees, sur le fond.
    INPUT_IS_LIST = True (singletons pris en [0]).

    >>> EXTENSION SUR PLEINE TOILE (v1.2) <<<
    'mask_adjust' n'est PLUS limite par le crop SAM3. Le masque de chaque crop est
    d'abord POSE sur une toile pleine image (noir partout ailleurs), puis grossi /
    retreci -> l'extension est bornee par l'IMAGE entiere, pas par le crop. On peut
    donc etendre une zone APRES coup sans relancer le KSampler. Le compositing se
    fait UNIQUEMENT dans le rectangle du crop : la ou le masque blanc deborde du
    crop, ce calque n'ecrit rien -> on voit le CALQUE DU DESSOUS (transparence),
    jamais l'image de base imposee. Grow/feather separables + downscale -> rapide
    meme a grande valeur (fini le "temps fou").

    HIERARCHIE ('order') :
      * 'list_first_on_top' : le 1er masque de la liste est AU-DESSUS ; le DERNIER
        est dessous. -> mets le masque de FOND (inverse de l'union) en DERNIER.
      * 'area_large_under'  : grandes zones dessous, petites dessus (auto).

    'checker' : contour colore de chaque zone pour verifier la superposition
    (toujours dessine ; non expose).

    v1.8 :
      * Batch : chaque image du batch est recomposee (avant : seule l'image 0,
        sans avertissement). Un crop / masque de batch 1 sert a toutes les images.
      * Base RGBA + crop RVB : l'alpha de la base est garde (avant : plantage
        sur le checker, et l'alpha aurait recu le canal rouge).
      * Bandes (crop qui deborde de l'image, x / y negatifs) : avant le resize
        retour, le contenu des bandes est remplace par le bord de l'image (image
        ET masque). Le gris des bandes ne deborde plus dans l'image (liseres) et
        le masque reste plein jusqu'au bord. Sans bande : inchange.
      * Arrondi au plus proche avant PIL (avant : troncature, -0,5 niveau).
      * Grow / feather calcules sur le rectangle du crop elargi de la portee
        (au lieu de toute la toile) : meme resultat au bit pres, bien plus
        rapide. Au-dela de 24 px (chemin sous-echantillonne), toute la toile,
        comme avant.
    """

    # Portee max du chemin plein resolution de _grow_shrink_fast / _feather_fast.
    _WINDOW_MAX_RADIUS = 24
    _use_window = True    # False : masque et checker sur toute la toile (reference des tests)

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "crops":      ("IMAGE",),
                "masks":      ("MASK",),
                "x":          ("INT", {"forceInput": True}),
                "y":          ("INT", {"forceInput": True}),
                "width":      ("INT", {"forceInput": True}),
                "height":     ("INT", {"forceInput": True}),
            },
            "optional": {
                "order":       (["list_first_on_top", "area_large_under"],),
                "mask_adjust": ("INT",   {"default": 0,   "min": -512, "max": 512}),
                "feather":     ("INT",   {"default": 8,   "min": 0, "max": 512}),
                "opacity":     ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "combined_mask", "checker")
    FUNCTION     = "assemble"
    CATEGORY     = "Aioli Nodes"

    @classmethod
    def _region_alpha(cls, m, x0, y0, x1, y1, cx0, cy0, H, W, mask_adjust, feather, opacity):
        """
        Alpha du calque : le masque (deja a la taille du crop) pose a sa place,
        grossi / retreci puis adouci. Renvoie (alpha, ox, oy) : alpha couvre la
        fenetre [oy:oy+h, ox:ox+w] de la toile, nulle ailleurs.

        Fenetre = rectangle du crop dans l'image + portee (mask_adjust + feather
        + 2), bornee par l'image. Au-dela de la portee, le calcul sur toute la
        toile ne donne que des zeros : le resultat est identique au bit pres
        (bord de l'image : meme padding ; bord interieur de la fenetre : zeros
        des deux cotes). Teste dans tests/test_assembler.py.
        """
        r_max = cls._WINDOW_MAX_RADIUS
        if cls._use_window and abs(mask_adjust) <= r_max and feather <= r_max:
            M = max(0, mask_adjust) + feather + 2
            ox, oy = max(0, x0 - M), max(0, y0 - M)
            ex, ey = min(W, x1 + M), min(H, y1 + M)
        else:
            ox, oy, ex, ey = 0, 0, W, H
        full_mask = torch.zeros((ey - oy, ex - ox), dtype=torch.float32)
        full_mask[y0 - oy:y1 - oy, x0 - ox:x1 - ox] = m[cy0:cy0 + (y1 - y0), cx0:cx0 + (x1 - x0)]
        full_mask = _grow_shrink_fast(full_mask, mask_adjust)
        alpha = _feather_fast(full_mask, feather) * opacity
        return alpha, ox, oy

    def assemble(self, base_image, crops, masks, x, y, width, height,
                 order=None, mask_adjust=None, feather=None, opacity=None):

        base_b = base_image[0]
        if base_b.dim() == 3:
            base_b = base_b.unsqueeze(0)
        base_b = base_b.float()
        Bb, H, W, C = base_b.shape

        order         = (order or ["list_first_on_top"])[0]
        mask_adjust   = int((mask_adjust or [0])[0])
        feather       = int((feather or [8])[0])
        opacity       = float((opacity or [1.0])[0])

        n = min(len(crops), len(masks), len(x), len(y), len(width), len(height))

        def as_batch(t, dims):
            t = t.float()
            return t.unsqueeze(0) if t.dim() == dims else t

        crop_b = [as_batch(crops[i], 3) for i in range(n)]
        mask_b = [as_batch(masks[i], 2) for i in range(n)]
        sizes = [Bb] + [c.shape[0] for c in crop_b] + [m.shape[0] for m in mask_b]
        B = max(sizes)
        if B > 1:
            print(f"[BBoxMultipleAssembler] batch {B}")
            if any(s not in (1, B) for s in sizes):
                print(f"[BBoxMultipleAssembler] WARNING : batchs incoherents {sorted(set(sizes))} "
                      f"-> le dernier element est repete jusqu'a {B}")

        idx = list(range(n))
        if order == "area_large_under":
            idx.sort(key=lambda i: int(width[i]) * int(height[i]), reverse=True)
        else:  # list_first_on_top -> on peint du dernier au premier
            idx = list(reversed(idx))

        # Par calque : geometrie, puis alpha (une fois par masque distinct).
        layers = []
        for i in idx:
            wi, hi = int(width[i]), int(height[i])
            xi, yi = int(x[i]), int(y[i])
            # rectangle du crop reellement present dans l'image (= la seule zone
            # ou ce calque possede des pixels enhances)
            x0, y0 = max(xi, 0), max(yi, 0)
            x1, y1 = min(xi + wi, W), min(yi + hi, H)
            if x1 <= x0 or y1 <= y0:
                continue
            cx0, cy0 = x0 - xi, y0 - yi
            cx1, cy1 = cx0 + (x1 - x0), cy0 + (y1 - y0)
            alphas = {}
            for bm in range(mask_b[i].shape[0]):
                m = mask_b[i][bm]
                if m.shape[0] != hi or m.shape[1] != wi:
                    m = self._prepare(m, wi, hi, cx0, cy0, cx1, cy1, _resize_mask)
                alphas[bm] = self._region_alpha(m, x0, y0, x1, y1, cx0, cy0, H, W,
                                                mask_adjust, feather, opacity)
            layers.append((i, wi, hi, x0, y0, x1, y1, cx0, cy0, cx1, cy1, alphas))

        prepared = {}

        def crop_for(i, cb, wi, hi, cx0, cy0, cx1, cy1):
            """Crop i, element cb du batch, a la taille du crop (une fois par (i, cb))."""
            key = (i, cb)
            if key not in prepared:
                crop = crop_b[i][cb]
                cc = crop.shape[2]
                if cc > C:
                    crop = crop[..., :C]
                elif cc == 1 and C > 1:
                    crop = crop.repeat(1, 1, min(C, 3))
                if crop.shape[0] != hi or crop.shape[1] != wi:
                    crop = self._prepare(crop, wi, hi, cx0, cy0, cx1, cy1, _resize_img)
                prepared[key] = crop
            return prepared[key]

        out_img, out_comb, out_check = [], [], []
        for b in range(B):
            canvas = base_b[min(b, Bb - 1)].clone()
            combined = torch.zeros((H, W), dtype=torch.float32)
            outlines = []
            for (i, wi, hi, x0, y0, x1, y1, cx0, cy0, cx1, cy1, alphas) in layers:
                crop = crop_for(i, min(b, crop_b[i].shape[0] - 1), wi, hi, cx0, cy0, cx1, cy1)
                alpha, ox, oy = alphas[min(b, len(alphas) - 1)]

                # COMPOSITE UNIQUEMENT dans le rectangle du crop.
                #   Hors du crop : on n'ecrit rien -> le calque DESSOUS (hierarchie)
                #   reste visible = "transparence". Pas de pixels de base imposes.
                a   = alpha[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
                src = crop[cy0:cy1, cx0:cx1, :]
                dst = canvas[y0:y1, x0:x1, :]
                if src.shape[2] < C:
                    # ex. crop RVB (VAE Decode) sur base RGBA : les canaux absents
                    # (l'alpha) restent ceux de la base. Avant : alpha = rouge.
                    src = torch.cat([src, dst[..., src.shape[2]:]], dim=-1)
                canvas[y0:y1, x0:x1, :] = src * a.unsqueeze(-1) + dst * (1.0 - a.unsqueeze(-1))
                combined[y0:y1, x0:x1] = torch.maximum(combined[y0:y1, x0:x1], a)
                outlines.append((a, x0, y0, x1, y1, i))

            # 'checker' : contour colore de chaque zone (toujours dessine)
            checker = canvas.clone()
            for (a, x0, y0, x1, y1, i) in outlines:
                g = 3   # grow 2 px du contour + 1
                if self._use_window:
                    gx0, gy0 = max(0, x0 - g), max(0, y0 - g)
                    gx1, gy1 = min(W, x1 + g), min(H, y1 + g)
                else:
                    gx0, gy0, gx1, gy1 = 0, 0, W, H
                m_rect = torch.zeros((gy1 - gy0, gx1 - gx0), dtype=torch.float32)
                m_rect[y0 - gy0:y1 - gy0, x0 - gx0:x1 - gx0] = (a > 0.5).float()
                edge = (_grow_shrink_fast(m_rect, 2) - m_rect).clamp(0, 1).unsqueeze(-1)
                col  = torch.tensor(_PALETTE[i % len(_PALETTE)], dtype=torch.float32)
                if C != 3:
                    col = torch.cat([col, torch.ones(max(0, C - 3))])[:C]
                win = checker[gy0:gy1, gx0:gx1, :]
                checker[gy0:gy1, gx0:gx1, :] = win * (1.0 - edge) + col * edge

            out_img.append(canvas.clamp(0.0, 1.0))
            out_comb.append(combined)
            out_check.append(checker.clamp(0.0, 1.0))

        print(f"[BBoxMultipleAssembler] {n} calques, order={order}, mask_adjust={mask_adjust} "
              f"(borne par l'image), feather={feather} -> {W}x{H}")
        return (torch.stack(out_img), torch.stack(out_comb), torch.stack(out_check))

    @staticmethod
    def _prepare(t, wi, hi, cx0, cy0, cx1, cy1, resize):
        """
        Remet un crop / masque de la taille modele a la taille du crop (wi x hi).
        S'il deborde de l'image (bandes), le contenu des bandes est d'abord
        remplace par le bord de l'image, a l'echelle modele : le resize ne
        melange plus le gris (ou ce que le modele y a peint) aux pixels de bord.
        Les bandes ne sont jamais recollees (hors de l'image de base).
        """
        h, w = t.shape[0], t.shape[1]
        if cx0 > 0 or cy0 > 0 or cx1 < wi or cy1 < hi:
            sx, sy = w / wi, h / hi
            mx0, mx1 = int(round(cx0 * sx)), int(round(cx1 * sx))
            my0, my1 = int(round(cy0 * sy)), int(round(cy1 * sy))
            mx0, my0 = min(max(mx0, 0), w - 1), min(max(my0, 0), h - 1)
            mx1, my1 = max(min(mx1, w), mx0 + 1), max(min(my1, h), my0 + 1)
            t = _replicate_outside(t, mx0, my0, mx1, my1)
        return resize(t, wi, hi)
