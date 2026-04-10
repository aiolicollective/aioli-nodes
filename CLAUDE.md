# CLAUDE.md — Aioli Nodes project context

> This file is read automatically by Claude Code at the start of each session.
> Update it at the end of each session to preserve important decisions and context.

---

## Project overview

**aioli-nodes** — Custom nodes for ComfyUI, published on the Comfy Registry.
- Publisher: `aiolicollective`
- Registry: `registry.comfy.org/nodes/aioli-nodes`
- GitHub: `https://github.com/aiolicollective/aioli-nodes`

**Versioning workflow:**
1. Edit code in `nodes/`
2. Bump version in `pyproject.toml` (semantic versioning: patch for fixes, minor for new features)
3. The GitHub Action `.github/workflows/publish.yml` auto-publishes to Comfy Registry on every push to `pyproject.toml`

**Current version: 1.0.10**

---

## Node inventory

### `nodes/bbox_fix.py` — `BBoxMultipleFix` — `📐 BBox Multiple Fix`
Core inpaint node. Plugs after `MaskBoundingBox+` (ComfyUI Essentials).

**Key design principles:**
- GCD algorithm ensures crop and resize target have the **exact same ratio** → pixel-perfect recomposition, 0% drift
- `avail_w / avail_h` anti-clamp: k is capped to available space around bbox center → crop never exceeds image source → clamp never modifies dims post-GCD → ratio guaranteed even at image borders
- All 9 outputs must remain in their current order — existing workflows depend on index positions

**Inputs:**
- `image`, `mask`, `x`, `y`, `width`, `height` — from MaskBoundingBox+
- `multiple` — `8/16/32/64` alignment
- `target` — `none/512/768/1024/1536/2048` — upscale OR downscale target
- `force_square` (bool, default False) — forces crop to 1:1 ratio (side = max(w,h))
- `force_target_downscale` (bool, default False) — if bbox > target, downscale GCD toward target instead of falling back to MAX_SIDE cap

**Outputs (order must not change):**
0. `image_cropped` IMAGE
1. `mask_cropped` MASK
2. `x` INT
3. `y` INT
4. `orig_width` INT — crop size in source (before any resize) → used for resize-back after VAE Decode
5. `orig_height` INT
6. `width` INT — final size after resize
7. `height` INT
8. `target_size` INT — numeric value of target dropdown (0 if "none") → connect to ImageResize+

**Scaling logic:**
| Case | Behaviour |
|------|-----------|
| bbox ≤ target | Upscale GCD toward target |
| bbox > target + `force_target_downscale=True` | Downscale GCD toward target |
| bbox > target + `force_target_downscale=False` | Fallback: round to multiple + cap MAX_SIDE (2048) |
| target = "none" + bbox > MAX_SIDE | Downscale GCD toward MAX_SIDE |

**Known edge case fixed in 1.0.10:**
When bbox center is near image border, GCD crop could exceed image dims → post-GCD clamp would break ratio. Fixed by capping k: `k = max(1, min(k, avail_w // a, avail_h // b))`.

---

### `nodes/inpaint_color_fix.py` — `InpaintColorFix` — `🎨 Inpaint Color Fix`
Post-VAE Decode color correction node. Plugs between VAE Decode and ImageResize+.

**Purpose:** Correct colorimetric drift introduced by the model on pixels that haven't creatively changed, while leaving truly generated pixels untouched.

**Key design:**
- Comparison in LAB color space (perceptual, D65) via Delta-E distance
- `mean_std` color match per channel, weighted by similarity mask
- No external dependencies — pure numpy + torch only

**Modes:**
- **No mask connected:** Delta-E auto-detects similar vs creative pixels
- **Mask connected (override):** Delta-E is bypassed entirely — mask drives correction directly

**Inputs:**
- `original_crop` — `image_cropped` from BBoxMultipleFix (before KSampler)
- `inpainted_crop` — IMAGE from VAE Decode
- `delta_e_threshold` — FLOAT, -1 = auto, 5–50 typical range
- `blend_strength` — FLOAT 0–1, strength of color correction on similar zones
- `feather_radius` — INT 0–64, gaussian blur on correction mask (pure torch, zero deps)
- `mask` (optional) — MASK override, bypasses Delta-E

**Outputs:**
- `image_corrected` — connect to ImageResize+
- `correction_mask` — debug: white = corrected, black = creative/intact

**Position in workflow:**
```
BBoxMultipleFix → KSampler → VAEDecode → InpaintColorFix → ImageResize+ → ImageCompositeMasked
```

---

### `nodes/ratio_outpaint.py` — `RatioOutpaintCalc` — `🖼️ Ratio Outpaint Calc`
Outpaint utility. Pads image to a standard aspect ratio and generates the mask.

---

## Comfy Registry publishing

**Trigger:** push to `pyproject.toml` on `main` branch
**Action:** `.github/workflows/publish.yml` using `Comfy-Org/publish-node-action@main`
**Secret required:** `REGISTRY_ACCESS_TOKEN` in repo secrets
**Important:** once a version is published on the Registry it is **immutable** — always bump before pushing

---

## Workflow example

The repo contains `examples/WF_Inpaint_aioli-nodes.json` — a complete Flux2Klein inpaint workflow packaged as a ComfyUI subgraph (`FLUX2KLEIN_INPAINT`).

Required custom nodes:
- Aioli Nodes (this repo)
- ComfyUI Essentials (`MaskBoundingBox+`, `ImageResize+`)
- ComfyUI KJNodes (`GrowMaskWithBlur`)
- rgthree-comfy (`Image Comparer`, optional)

Required models:
- `flux2/flux-2-klein-9b-fp8.safetensors`
- `flux2/flux2-vae.safetensors`
- `qwen_3_8b_fp8mixed.safetensors`

---

## Session update log

| Version | Date | Changes |
|---------|------|---------|
| 1.0.1 | 2026-03-08 | Initial Registry publication |
| 1.0.2 | 2026-03-12 | BBoxMultipleFix: cap crop at 2048px in fallback mode |
| 1.0.3 | 2026-03-12 | BBoxMultipleFix: downscale full zone instead of truncating |
| 1.0.4 | 2026-03-12 | BBoxMultipleFix: fix ratio preservation on downscale via GCD |
| 1.0.5 | 2026-03-14 | Add InpaintColorFix node (testing) |
| 1.0.6 | 2026-03-16 | InpaintColorFix: mask override mode + feather cleanup |
| 1.0.7 | 2026-03-16 | BBoxMultipleFix: add target_size INT output (slot 8) |
| 1.0.8 | 2026-04-03 | BBoxMultipleFix: add force_square + force_target_downscale booleans |
| 1.0.9 | 2026-04-03 | Intermediate (pre anti-clamp) |
| 1.0.10 | 2026-04-10 | BBoxMultipleFix: anti-clamp fix — 0% ratio drift at image borders |
