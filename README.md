# Aioli Nodes — ComfyUI Custom Node Suite

Two utility nodes for outpainting and inpainting in ComfyUI.

## Installation

### Via ComfyUI Manager (recommended)

**Install via Git URL:**
```
https://github.com/aiolicollective/aioli-nodes
```

> ℹ️ A submission to the [ComfyUI Registry](https://registry.comfy.org) is in progress so the nodes will soon be installable directly from the built-in manager.

### Manual installation

1. Copy the `aioli-nodes` folder into `ComfyUI/custom_nodes/`
2. Restart ComfyUI
3. The nodes appear under the **Aioli Nodes** category

No extra dependencies — only `math` (Python stdlib), `torch` and `Pillow` (already bundled with ComfyUI).

---

## 🖼️ Ratio Outpaint Calc

Prepares an image for outpainting to a standard aspect ratio.  
Automatically computes padding and generates the mask.

**Inputs**
| Parameter | Type | Description |
|-----------|------|-------------|
| image | IMAGE | Source image |
| ratio | dropdown | `none` · `1:1` · `4:5` · `5:4` · `3:4` · `4:3` · `16:9` · `9:16` |

**Outputs**
| Output | Type | Description |
|--------|------|-------------|
| image_padded | IMAGE | Image padded with neutral grey (0.5) |
| mask | MASK | Binary mask (0 = keep, 1 = generate) |

**Workflow**
```
Load Image → 🖼️ Ratio Outpaint Calc → VAE Encode (Inpaint) → KSampler
```

---

## 📐 BBox Multiple Fix

Plugs in right after **Mask Bounding Box** (ComfyUI Essentials).  
Rounds the crop to the chosen multiple and optionally upscales to a Flux-friendly resolution.

The node ensures the inpainted region stitches back **pixel-perfectly** onto the base image — no border artefacts, no alignment drift.

**Example**

![BBox Multiple Fix — inpaint example](examples/BBOX_fixe_inpaint.jpg)

*Left to right: base image → mask → inpaint result → before/after overlay.  
The edited region fits the original contours exactly.*

**Inputs**
| Parameter | Type | Description |
|-----------|------|-------------|
| image | IMAGE | Full source image (before crop) |
| mask | MASK | Full source mask (before crop) |
| x | INT | `x` output from Mask Bounding Box |
| y | INT | `y` output from Mask Bounding Box |
| width | INT | `width` output from Mask Bounding Box |
| height | INT | `height` output from Mask Bounding Box |
| multiple | dropdown | `8 (VAE)` · `32 (SD1.5)` · `64 (SDXL/Flux)` |
| target | dropdown | `none` · `512` · `768` · `1024` · `1536` · `2048` |

**Outputs**
| Output | Type | Description |
|--------|------|-------------|
| image_cropped | IMAGE | Cropped image (upscaled if target is set) |
| mask_cropped | MASK | Cropped mask (upscaled if target is set) |
| x | INT | x position for ImageCompositeMasked |
| y | INT | y position for ImageCompositeMasked |
| orig_width | INT | Crop width BEFORE upscale (use for resize-back) |
| orig_height | INT | Crop height BEFORE upscale (use for resize-back) |
| width | INT | Final width |
| height | INT | Final height |

**Workflow without upscale**
```
BBox Fix → VAE Encode → KSampler → VAE Decode → ImageCompositeMasked ← x, y
```

**Workflow with upscale**
```
BBox Fix → VAE Encode → KSampler → VAE Decode
  ↓ orig_width, orig_height              ↓
  ↓ x, y        → Image Resize ←────────┘
                       ↓
               ImageCompositeMasked ← x, y
```
