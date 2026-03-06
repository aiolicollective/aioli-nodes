# Aioli Nodes — ComfyUI Custom Node Suite

Deux nodes utilitaires pour l'outpainting et l'inpainting dans ComfyUI.

## Installation

1. Copie le dossier `aioli-nodes` dans `ComfyUI/custom_nodes/`
2. Redémarre ComfyUI
3. Les nodes apparaissent dans la catégorie **Aioli Nodes**

Ou via **ComfyUI Manager** → Install via Git URL :
```
https://github.com/aiolicollective/aioli-nodes
```

Aucune dépendance à installer — utilise uniquement `math` (Python stdlib), `torch` et `Pillow` (déjà dans ComfyUI).

---

## 🖼️ Ratio Outpaint Calc

Prépare une image pour l'outpainting vers un ratio standard.  
Calcule automatiquement le padding et génère le masque.

**Entrées**
| Paramètre | Type | Description |
|-----------|------|-------------|
| image | IMAGE | Image source |
| ratio | dropdown | `none` · `1:1` · `4:5` · `5:4` · `3:4` · `4:3` · `16:9` · `9:16` |

**Sorties**
| Sortie | Type | Description |
|--------|------|-------------|
| image_padded | IMAGE | Image paddée en gris neutre (0.5) |
| mask | MASK | Masque binaire (0=conserver, 1=générer) |

**Workflow**
```
Load Image → 🖼️ Ratio Outpaint Calc → VAE Encode (Inpaint) → KSampler
```

---

## 📐 BBox Multiple Fix

S'insère après **Mask Bounding Box** (ComfyUI Essentials).  
Arrondit le crop au multiple choisi et upscale optionnellement vers une résolution Flux.

**Entrées**
| Paramètre | Type | Description |
|-----------|------|-------------|
| image | IMAGE | Image source complète (avant crop) |
| mask | MASK | Masque source complet (avant crop) |
| x | INT | Sortie x de Mask Bounding Box |
| y | INT | Sortie y de Mask Bounding Box |
| width | INT | Sortie width de Mask Bounding Box |
| height | INT | Sortie height de Mask Bounding Box |
| multiple | dropdown | `8 (VAE)` · `32 (SD1.5)` · `64 (SDXL/Flux)` |
| target | dropdown | `none` · `512` · `768` · `1024` · `1536` · `2048` |

**Sorties**
| Sortie | Type | Description |
|--------|------|-------------|
| image_cropped | IMAGE | Image croppée (upscalée si target) |
| mask_cropped | MASK | Masque croppé (upscalé si target) |
| x | INT | Position x pour ImageCompositeMasked |
| y | INT | Position y pour ImageCompositeMasked |
| orig_width | INT | Largeur crop AVANT upscale (resize retour) |
| orig_height | INT | Hauteur crop AVANT upscale (resize retour) |
| width | INT | Largeur finale |
| height | INT | Hauteur finale |

**Workflow sans upscale**
```
BBox Fix → VAE Encode → KSampler → VAE Decode → ImageCompositeMasked ← x, y
```

**Workflow avec upscale**
```
BBox Fix → VAE Encode → KSampler → VAE Decode
  ↓ orig_width, orig_height              ↓
  ↓ x, y        → Image Resize ←────────┘
                       ↓
               ImageCompositeMasked ← x, y
```
