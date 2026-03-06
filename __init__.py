from .nodes.ratio_outpaint import RatioOutpaintCalc
from .nodes.bbox_fix        import BBoxMultipleFix

NODE_CLASS_MAPPINGS = {
    "RatioOutpaintCalc": RatioOutpaintCalc,
    "BBoxMultipleFix":   BBoxMultipleFix,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RatioOutpaintCalc": "🖼️ Ratio Outpaint Calc",
    "BBoxMultipleFix":   "📐 BBox Multiple Fix",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
