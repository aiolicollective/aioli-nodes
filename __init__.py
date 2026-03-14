from .nodes.ratio_outpaint    import RatioOutpaintCalc
from .nodes.bbox_fix          import BBoxMultipleFix
from .nodes.inpaint_color_fix import InpaintColorFix

NODE_CLASS_MAPPINGS = {
    "RatioOutpaintCalc": RatioOutpaintCalc,
    "BBoxMultipleFix":   BBoxMultipleFix,
    "InpaintColorFix":   InpaintColorFix,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RatioOutpaintCalc": "🖼️ Ratio Outpaint Calc",
    "BBoxMultipleFix":   "📐 BBox Multiple Fix",
    "InpaintColorFix":   "🎨 Inpaint Color Fix",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
