from .nodes.ratio_outpaint         import RatioOutpaintCalc
from .nodes.mask_bbox_v2           import AioliMaskBBoxV2
from .nodes.inpaint_color_fix      import InpaintColorFix
from .nodes.bbox_assembler         import BBoxMultipleAssembler
from .nodes.regional_conditioning  import RegionalMaskConditioning
from .nodes.region_mask_list       import RegionMaskList
from .nodes.mask_split_regions     import MaskSplitRegions
from .nodes.region_preview         import RegionPreview

# Legacy — gardes enregistres pour que les anciens workflows continuent de se
# charger et de tourner a l'identique. Ne pas retirer.
#   AioliMaskBBox   : remplace par AioliMaskBBoxV2 en v1.7.0
#   BBoxMultipleFix : remplace par AioliMaskBBox en v1.6.0
from .nodes.mask_bbox              import AioliMaskBBox
from .nodes.bbox_fix               import BBoxMultipleFix

NODE_CLASS_MAPPINGS = {
    "RatioOutpaintCalc":        RatioOutpaintCalc,
    "AioliMaskBBoxV2":          AioliMaskBBoxV2,
    "InpaintColorFix":          InpaintColorFix,
    "BBoxMultipleAssembler":    BBoxMultipleAssembler,
    "RegionalMaskConditioning": RegionalMaskConditioning,
    "RegionMaskList":           RegionMaskList,
    "MaskSplitRegions":         MaskSplitRegions,
    "RegionPreview":            RegionPreview,

    # Legacy
    "AioliMaskBBox":            AioliMaskBBox,
    "BBoxMultipleFix":          BBoxMultipleFix,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RatioOutpaintCalc":        "🖼️ Ratio Outpaint Calc",
    "AioliMaskBBoxV2":          "📐 Aioli Mask BBox V2",
    "InpaintColorFix":          "🎨 Inpaint Color Fix",
    "BBoxMultipleAssembler":    "🧩 BBox Multiple Assembler",
    "RegionalMaskConditioning": "🗺️ Regional Mask Conditioning",
    "RegionMaskList":           "🧱 Region Mask List (+background)",
    "MaskSplitRegions":         "✂️ Mask Split Regions (manual multi)",
    "RegionPreview":            "👁 Region Preview",

    # Legacy
    "AioliMaskBBox":            "📐 Aioli Mask BBox (legacy)",
    "BBoxMultipleFix":          "📐 BBox Multiple Fix (legacy)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
