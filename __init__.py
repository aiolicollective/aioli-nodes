from .nodes.ratio_outpaint         import RatioOutpaintCalc
from .nodes.bbox_fix               import BBoxMultipleFix
from .nodes.inpaint_color_fix      import InpaintColorFix
from .nodes.bbox_assembler         import BBoxMultipleAssembler
from .nodes.regional_conditioning  import RegionalMaskConditioning

NODE_CLASS_MAPPINGS = {
    "RatioOutpaintCalc":        RatioOutpaintCalc,
    "BBoxMultipleFix":          BBoxMultipleFix,
    "InpaintColorFix":          InpaintColorFix,
    "BBoxMultipleAssembler":    BBoxMultipleAssembler,
    "RegionalMaskConditioning": RegionalMaskConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RatioOutpaintCalc":        "🖼️ Ratio Outpaint Calc",
    "BBoxMultipleFix":          "📐 BBox Multiple Fix",
    "InpaintColorFix":          "🎨 Inpaint Color Fix",
    "BBoxMultipleAssembler":    "🧩 BBox Multiple Assembler",
    "RegionalMaskConditioning": "🗺️ Regional Mask Conditioning",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
