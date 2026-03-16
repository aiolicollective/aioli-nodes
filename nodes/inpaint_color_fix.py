import math
import torch
import torch.nn.functional as F
import numpy as np


class InpaintColorFix:
    """
    S'insère après VAE Decode, avant ImageResize+ / ImageCompositeMasked.

    Sans mask externe :
      Compare original_crop et inpainted_crop en espace LAB via Delta-E.
      Pixels similaires (Delta-E < threshold) → color match appliqué.
      Pixels créatifs   (Delta-E > threshold) → inpainted intact.

    Avec mask externe (override) :
      Le Delta-E est ignoré. Le masque fourni pilote directement la correction.
      Blanc = color match appliqué / Noir = inpainted intact.

    feather_radius : gaussian blur (pure torch) du masque actif —
                     transition douce entre zones corrigées/intactes.

    Entrées :
      original_crop     → image_cropped depuis BBoxMultipleFix
      inpainted_crop    → IMAGE depuis VAE Decode
      delta_e_threshold → seuil créatif/similaire (-1 = auto) — ignoré si mask branché
      blend_strength    → force du color match (0=aucun, 1=total)
      feather_radius    → rayon du blur du masque actif (0=désactivé)
      mask (opt.)       → MASK override : bypass Delta-E, contrôle direct

    Sorties :
      image_corrected  → crop corrigé à brancher sur ImageResize+
      correction_mask  → masque debug (blanc = corrigé, noir = créatif intact)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_crop":     ("IMAGE",),
                "inpainted_crop":    ("IMAGE",),
                "delta_e_threshold": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 100.0, "step": 1.0,
                }),
                "blend_strength":    ("FLOAT", {
                    "default": 1.0,  "min": 0.0,  "max": 1.0,   "step": 0.05,
                }),
                "feather_radius":    ("INT", {
                    "default": 0,    "min": 0,     "max": 64,    "step": 1,
                }),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES  = ("IMAGE", "MASK")
    RETURN_NAMES  = ("image_corrected", "correction_mask")
    FUNCTION      = "fix"
    CATEGORY      = "Aioli Nodes"

    # ------------------------------------------------------------------
    # RGB <-> LAB (pure numpy, D65)
    # ------------------------------------------------------------------

    def _rgb_to_lab(self, rgb: np.ndarray) -> np.ndarray:
        """rgb : float32 [H,W,3] 0-1  →  LAB float32 [H,W,3]"""
        lin = np.where(
            rgb <= 0.04045,
            rgb / 12.92,
            ((rgb + 0.055) / 1.055) ** 2.4,
        ).astype(np.float32)
        M = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ], dtype=np.float32)
        xyz = (lin.reshape(-1, 3) @ M.T).reshape(lin.shape)
        xyz /= np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)

        eps = (6 / 29) ** 3
        kap = 1 / (3 * (6 / 29) ** 2)
        f   = np.where(xyz > eps, np.cbrt(xyz), kap * xyz + 4 / 29)
        L   = 116 * f[..., 1] - 16
        a   = 500 * (f[..., 0] - f[..., 1])
        b   = 200 * (f[..., 1] - f[..., 2])
        return np.stack([L, a, b], axis=-1).astype(np.float32)

    def _lab_to_rgb(self, lab: np.ndarray) -> np.ndarray:
        """LAB float32 [H,W,3]  →  rgb float32 [H,W,3] 0-1"""
        fy   = (lab[..., 0] + 16) / 116
        fx   = lab[..., 1] / 500 + fy
        fz   = fy - lab[..., 2] / 200
        eps  = 6 / 29
        fxyz = np.stack([fx, fy, fz], axis=-1)
        xyz  = np.where(
            fxyz > eps,
            fxyz ** 3,
            3 * (6 / 29) ** 2 * (fxyz - 4 / 29),
        ).astype(np.float32)
        xyz *= np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)

        M_inv = np.array([
            [ 3.2404542, -1.5371385, -0.4985314],
            [-0.9692660,  1.8760108,  0.0415560],
            [ 0.0556434, -0.2040259,  1.0572252],
        ], dtype=np.float32)
        lin = (xyz.reshape(-1, 3) @ M_inv.T).reshape(xyz.shape)
        rgb = np.where(
            lin <= 0.0031308,
            12.92 * lin,
            1.055 * np.power(np.clip(lin, 0, None), 1 / 2.4) - 0.055,
        )
        return np.clip(rgb, 0, 1).astype(np.float32)

    # ------------------------------------------------------------------
    # Gaussian blur du masque (pure torch, zéro dépendance externe)
    # ------------------------------------------------------------------

    def _feather_mask(self, mask: np.ndarray, radius: int) -> np.ndarray:
        """Gaussian blur du masque via torch.nn.functional.conv2d."""
        if radius <= 0:
            return mask
        sigma  = radius / 2.0
        size   = radius * 2 + 1
        coords = torch.arange(size, dtype=torch.float32) - radius
        gauss  = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)
        kernel = (kernel / kernel.sum()).unsqueeze(0).unsqueeze(0)  # [1,1,k,k]
        t      = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        return F.conv2d(t, kernel, padding=radius).squeeze().numpy()

    # ------------------------------------------------------------------
    # Color match mean_std pondéré (LAB)
    # ------------------------------------------------------------------

    def _match_mean_std(self, src_lab: np.ndarray, ref_lab: np.ndarray,
                        weight: np.ndarray) -> np.ndarray:
        """Recale mean/std de src vers ref, canal par canal, pondéré par weight."""
        result = src_lab.copy()
        for c in range(3):
            s  = src_lab[..., c]
            r  = ref_lab[..., c]
            ws = weight.sum() + 1e-8
            s_mean = (s * weight).sum() / ws
            r_mean = (r * weight).sum() / ws
            s_std  = math.sqrt(((s - s_mean) ** 2 * weight).sum() / ws + 1e-8)
            r_std  = math.sqrt(((r - r_mean) ** 2 * weight).sum() / ws + 1e-8)
            result[..., c] = (s - s_mean) * (r_std / (s_std + 1e-8)) + r_mean
        return result

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def fix(self, original_crop, inpainted_crop,
            delta_e_threshold, blend_strength, feather_radius,
            mask=None):

        orig_np = original_crop[0].cpu().float().numpy()   # [H,W,3] 0-1
        inp_np  = inpainted_crop[0].cpu().float().numpy()  # [H,W,3] 0-1

        # Assure même taille
        if orig_np.shape != inp_np.shape:
            from PIL import Image
            H, W    = inp_np.shape[:2]
            orig_np = np.array(
                Image.fromarray((orig_np * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)
            ).astype(np.float32) / 255.0

        H, W = orig_np.shape[:2]

        # ------------------------------------------------------------------
        # Masque actif : override (mask branché) ou Delta-E
        # ------------------------------------------------------------------
        if mask is not None:
            # OVERRIDE — Delta-E ignoré, le masque externe commande
            active_mask = mask[0].cpu().float().numpy()
            if active_mask.shape != (H, W):
                from PIL import Image
                active_mask = np.array(
                    Image.fromarray((active_mask * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)
                ).astype(np.float32) / 255.0
            print(f"[InpaintColorFix] mode=mask_override  feather={feather_radius}px")

        else:
            # Delta-E — similarité colorimétrique LAB
            orig_lab = self._rgb_to_lab(orig_np)
            inp_lab  = self._rgb_to_lab(inp_np)
            diff     = orig_lab - inp_lab
            diff[..., 0] *= 0.7   # pondère L vs chrominance
            delta_e  = np.sqrt((diff ** 2).sum(axis=-1))

            if delta_e_threshold < 0:
                p50       = float(np.percentile(delta_e, 50))
                p75       = float(np.percentile(delta_e, 75))
                threshold = float(np.clip(p75 + (p75 - p50) * 0.5, 4.0, 60.0))
                print(f"[InpaintColorFix] mode=delta_e  auto threshold={threshold:.1f}  "
                      f"(p50={p50:.1f}  p75={p75:.1f})  feather={feather_radius}px")
            else:
                threshold = delta_e_threshold
                print(f"[InpaintColorFix] mode=delta_e  threshold={threshold:.1f}  "
                      f"feather={feather_radius}px")

            active_mask = (delta_e <= threshold).astype(np.float32)

        # Feathering du masque actif (gaussian blur pure torch)
        active_mask = np.clip(self._feather_mask(active_mask, feather_radius), 0.0, 1.0)

        corrected_pct = 100.0 * (active_mask >= 0.5).sum() / (H * W)
        print(f"[InpaintColorFix] corrected={corrected_pct:.1f}%  "
              f"untouched={100-corrected_pct:.1f}%")

        # Color match mean_std en LAB
        orig_lab = self._rgb_to_lab(orig_np)
        inp_lab  = self._rgb_to_lab(inp_np)
        corrected_lab = self._match_mean_std(inp_lab, orig_lab, active_mask)

        # Blend final pondéré par masque × blend_strength
        weight3   = (active_mask * blend_strength)[..., np.newaxis]
        final_lab = inp_lab + weight3 * (corrected_lab - inp_lab)

        result = self._lab_to_rgb(final_lab)

        corr_mask_out = torch.from_numpy(
            np.clip(active_mask * blend_strength, 0, 1).astype(np.float32)
        ).unsqueeze(0)

        return (
            torch.from_numpy(result).unsqueeze(0),
            corr_mask_out,
        )
