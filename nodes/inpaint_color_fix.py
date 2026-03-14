import math
import torch
import numpy as np


COLOR_MATCH_MODES = ["mean_std", "histogram"]


class InpaintColorFix:
    """
    S'insère après VAE Decode, avant ImageResize+ / ImageCompositeMasked.

    Compare original_crop (avant KSampler) et inpainted_crop (après VAE Decode)
    en espace LAB pixel par pixel via Delta-E.

    - Pixels similaires (Delta-E < threshold) : color match appliqué
      pour corriger la dérive colorimétrique introduite par la génération.
    - Pixels créatifs   (Delta-E > threshold) : inpainted intact, pas de correction.

    Le masque de correction est exposé en sortie pour debug.

    Entrées :
      original_crop   → image_cropped depuis BBoxMultipleFix
      inpainted_crop  → IMAGE depuis VAE Decode
      delta_e_threshold → seuil de détection créatif/similaire (-1 = auto)
      blend_strength  → force du color match sur zones similaires (0=aucun, 1=total)
      color_match_mode → mean_std ou histogram

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
                "color_match_mode":  (COLOR_MATCH_MODES,),
            }
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
        f = np.where(xyz > eps, np.cbrt(xyz), kap * xyz + 4 / 29)
        L = 116 * f[..., 1] - 16
        a = 500 * (f[..., 0] - f[..., 1])
        b = 200 * (f[..., 1] - f[..., 2])
        return np.stack([L, a, b], axis=-1).astype(np.float32)

    def _lab_to_rgb(self, lab: np.ndarray) -> np.ndarray:
        """LAB float32 [H,W,3]  →  rgb float32 [H,W,3] 0-1"""
        fy = (lab[..., 0] + 16) / 116
        fx = lab[..., 1] / 500 + fy
        fz = fy - lab[..., 2] / 200

        eps = 6 / 29
        xyz = np.where(
            np.stack([fx, fy, fz], axis=-1) > eps,
            np.stack([fx, fy, fz], axis=-1) ** 3,
            3 * (6 / 29) ** 2 * (np.stack([fx, fy, fz], axis=-1) - 4 / 29),
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
    # Color match helpers
    # ------------------------------------------------------------------

    def _match_mean_std(self, src_lab: np.ndarray, ref_lab: np.ndarray,
                        weight: np.ndarray) -> np.ndarray:
        """
        Recale mean/std de src vers ref canal par canal (L, a, b),
        pondéré par weight (float32 [H,W], valeurs 0-1 = zones à corriger).
        """
        result = src_lab.copy()
        for c in range(3):
            s = src_lab[..., c]
            r = ref_lab[..., c]
            w = weight
            ws = w.sum() + 1e-8

            s_mean = (s * w).sum() / ws
            r_mean = (r * w).sum() / ws
            s_std  = math.sqrt(((s - s_mean) ** 2 * w).sum() / ws + 1e-8)
            r_std  = math.sqrt(((r - r_mean) ** 2 * w).sum() / ws + 1e-8)

            scale = r_std / (s_std + 1e-8)
            corrected = (s - s_mean) * scale + r_mean
            result[..., c] = corrected
        return result

    def _match_histogram(self, src_lab: np.ndarray, ref_lab: np.ndarray,
                         weight: np.ndarray) -> np.ndarray:
        """
        Histogram matching canal par canal, pondéré par weight.
        Pour chaque canal, on construit les CDFs sur les pixels similaires
        et on mappe src → ref.
        """
        result = src_lab.copy()
        bins = 256
        for c in range(3):
            s = src_lab[..., c].ravel()
            r = ref_lab[..., c].ravel()
            w = weight.ravel()

            vmin = min(s.min(), r.min())
            vmax = max(s.max(), r.max()) + 1e-6

            # CDF source pondérée
            s_hist, edges = np.histogram(s, bins=bins, range=(vmin, vmax), weights=w)
            s_cdf = np.cumsum(s_hist).astype(np.float32)
            s_cdf /= (s_cdf[-1] + 1e-8)

            # CDF référence pondérée
            r_hist, _ = np.histogram(r, bins=bins, range=(vmin, vmax), weights=w)
            r_cdf = np.cumsum(r_hist).astype(np.float32)
            r_cdf /= (r_cdf[-1] + 1e-8)

            # Mapping s → r via CDFs
            bin_centers = (edges[:-1] + edges[1:]) / 2
            s_idx = np.searchsorted(s_cdf, np.clip(
                np.interp(s, bin_centers, s_cdf), 0, 1 - 1e-8
            ))
            s_idx = np.clip(s_idx, 0, bins - 1)
            corrected = bin_centers[s_idx]
            result[..., c] = corrected.reshape(src_lab.shape[:2])
        return result

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def fix(self, original_crop, inpainted_crop,
            delta_e_threshold, blend_strength, color_match_mode):

        orig_np = original_crop[0].cpu().float().numpy()     # [H,W,3] 0-1
        inp_np  = inpainted_crop[0].cpu().float().numpy()    # [H,W,3] 0-1

        # Assure même taille (au cas où)
        if orig_np.shape != inp_np.shape:
            from PIL import Image
            H, W = inp_np.shape[:2]
            pil = Image.fromarray((orig_np * 255).astype(np.uint8))
            orig_np = np.array(pil.resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0

        H, W = orig_np.shape[:2]

        # Conversion LAB
        orig_lab = self._rgb_to_lab(orig_np)
        inp_lab  = self._rgb_to_lab(inp_np)

        # Delta-E perceptuel (distance euclidienne en LAB)
        diff = orig_lab - inp_lab
        diff[..., 0] *= 0.7   # pondère légèrement L vs chrominance
        delta_e = np.sqrt((diff ** 2).sum(axis=-1))   # [H,W]

        # Seuil auto si -1
        if delta_e_threshold < 0:
            p50 = float(np.percentile(delta_e, 50))
            p75 = float(np.percentile(delta_e, 75))
            threshold = p75 + (p75 - p50) * 0.5
            threshold = float(np.clip(threshold, 4.0, 60.0))
            print(f"[InpaintColorFix] auto threshold : {threshold:.1f} "
                  f"(p50={p50:.1f} p75={p75:.1f})")
        else:
            threshold = delta_e_threshold

        # Masque de correction : 1 = similaire (à corriger), 0 = créatif (intact)
        similar_mask = (delta_e <= threshold).astype(np.float32)   # [H,W]

        changed_pct = 100.0 * (similar_mask < 0.5).sum() / (H * W)
        print(f"[InpaintColorFix] threshold={threshold:.1f}  "
              f"creative={changed_pct:.1f}%  similar={100-changed_pct:.1f}%")

        # Color match sur les pixels similaires
        if color_match_mode == "mean_std":
            corrected_lab = self._match_mean_std(inp_lab, orig_lab, similar_mask)
        else:
            corrected_lab = self._match_histogram(inp_lab, orig_lab, similar_mask)

        # Blend : blend_strength contrôle la force de la correction
        blended_lab = inp_lab + similar_mask[..., np.newaxis] * blend_strength * (corrected_lab - inp_lab)

        # Pixels créatifs → inpainted intact
        creative_mask = (1.0 - similar_mask)[..., np.newaxis]
        final_lab = blended_lab * (1.0 - creative_mask) + inp_lab * creative_mask

        # Retour RGB
        result = self._lab_to_rgb(final_lab)

        # Correction mask pour debug (blanc = corrigé)
        corr_mask_out = torch.from_numpy(
            (similar_mask * blend_strength).astype(np.float32)
        ).unsqueeze(0)   # [1,H,W]

        return (
            torch.from_numpy(result).unsqueeze(0),
            corr_mask_out,
        )
