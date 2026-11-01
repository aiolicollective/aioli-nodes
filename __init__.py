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
    "AioliMaskBBoxV2":          "📐 Aioli Mask BBox",
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

# v1.8 : panneau Resolution (API V3, widget JS dans web/). Si l'API V3 ou le
# serveur ne sont pas disponibles (ComfyUI ancien), le reste du pack charge.
WEB_DIRECTORY = "./web"
try:
    from .nodes.resolution_panel import AioliResolutionPanel
    from .nodes import resolution_routes  # noqa: F401  (route /aioli/resolution_plan)
    NODE_CLASS_MAPPINGS["AioliResolutionPanel"] = AioliResolutionPanel
    NODE_DISPLAY_NAME_MAPPINGS["AioliResolutionPanel"] = "📏 Aioli Resolution Panel"
except Exception as e:  # pragma: no cover
    print(f"[aioli-nodes] Resolution Panel indisponible : {e}")

# v1.8 : comparateur video A / B (API V3, widget JS dans web/).
try:
    from .nodes.video_compare import AioliVideoCompare
    NODE_CLASS_MAPPINGS["AioliVideoCompare"] = AioliVideoCompare
    NODE_DISPLAY_NAME_MAPPINGS["AioliVideoCompare"] = "🎞️ Aioli Video Compare"
except Exception as e:  # pragma: no cover
    print(f"[aioli-nodes] Video Compare indisponible : {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
