from .nodes.ratio_outpaint         import RatioOutpaintCalc
from .nodes.bbox_fix               import BBoxMultipleFix
from .nodes.inpaint_color_fix      import InpaintColorFix
from .nodes.bbox_assembler         import BBoxMultipleAssembler
from .nodes.regional_conditioning  import RegionalMaskConditioning
from .nodes.region_mask_list       import RegionMaskList
from .nodes.mask_split_regions     import MaskSplitRegions
from .nodes.region_preview         import RegionPreview

NODE_CLASS_MAPPINGS = {
    "RatioOutpaintCalc":        RatioOutpaintCalc,
    "BBoxMultipleFix":          BBoxMultipleFix,
    "InpaintColorFix":          InpaintColorFix,
    "BBoxMultipleAssembler":    BBoxMultipleAssembler,
    "RegionalMaskConditioning": RegionalMaskConditioning,
    "RegionMaskList":           RegionMaskList,
    "MaskSplitRegions":         MaskSplitRegions,
    "RegionPreview":            RegionPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RatioOutpaintCalc":        "🖼️ Ratio Outpaint Calc",
    "BBoxMultipleFix":          "📐 BBox Multiple Fix",
    "InpaintColorFix":          "🎨 Inpaint Color Fix",
    "BBoxMultipleAssembler":    "🧩 BBox Multiple Assembler",
    "RegionalMaskConditioning": "🗺️ Regional Mask Conditioning",
    "RegionMaskList":           "🧱 Region Mask List (+background)",
    "MaskSplitRegions":         "✂️ Mask Split Regions (manual multi)",
    "RegionPreview":            "👁 Region Preview",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
