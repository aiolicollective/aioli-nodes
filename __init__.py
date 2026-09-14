from .nodes.ratio_outpaint         import RatioOutpaintCalc
from .nodes.mask_bbox              import AioliMaskBBox
from .nodes.inpaint_color_fix      import InpaintColorFix
from .nodes.bbox_assembler         import BBoxMultipleAssembler
from .nodes.regional_conditioning  import RegionalMaskConditioning
from .nodes.region_mask_list       import RegionMaskList
from .nodes.mask_split_regions     import MaskSplitRegions
from .nodes.region_preview         import RegionPreview

# Legacy — remplace par AioliMaskBBox, garde enregistre pour que les anciens
# workflows continuent de se charger. Ne pas retirer.
from .nodes.bbox_fix               import BBoxMultipleFix

NODE_CLASS_MAPPINGS = {
    "RatioOutpaintCalc":        RatioOutpaintCalc,
    "AioliMaskBBox":            AioliMaskBBox,
    "InpaintColorFix":          InpaintColorFix,
    "BBoxMultipleAssembler":    BBoxMultipleAssembler,
    "RegionalMaskConditioning": RegionalMaskConditioning,
    "RegionMaskList":           RegionMaskList,
    "MaskSplitRegions":         MaskSplitRegions,
    "RegionPreview":            RegionPreview,

    # Legacy
    "BBoxMultipleFix":          BBoxMultipleFix,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RatioOutpaintCalc":        "🖼️ Ratio Outpaint Calc",
    "AioliMaskBBox":            "📐 Aioli Mask BBox",
    "InpaintColorFix":          "🎨 Inpaint Color Fix",
    "BBoxMultipleAssembler":    "🧩 BBox Multiple Assembler",
    "RegionalMaskConditioning": "🗺️ Regional Mask Conditioning",
    "RegionMaskList":           "🧱 Region Mask List (+background)",
    "MaskSplitRegions":         "✂️ Mask Split Regions (manual multi)",
    "RegionPreview":            "👁 Region Preview",

    # Legacy
    "BBoxMultipleFix":          "📐 BBox Multiple Fix (legacy)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
