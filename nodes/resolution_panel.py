"""
📏 Aioli Resolution Panel (cle "AioliResolutionPanel", API V3).

Panneau de reglage de la resolution des wf V2 : il porte SCALE, OUTPAINT RATIO,
REGENERATE WHOLE IMAGE, RESOLUTION, multiple, SQUARE, DOWNSCALE TO RESOLUTION,
padding, CONTEXT, les sort pour les entrees du subgraph, et dessine sur le node
la chaine entiere a l'echelle : source -> SCALE -> outpaint -> par region bbox,
crop, taille modele, bandes, up / downscale.

Avant run : le schema suit les reglages en direct (web/aioli_resolution_panel.js,
route /aioli/resolution_plan). Apres run : valeurs exactes de l'execution.
Le calcul est celui des nodes (nodes/resolution_plan.py), pas une copie.
"""

from comfy_api.latest import io

from .bbox_core_v2 import MAX_RESOLUTION, MULTIPLES
from .mask_bbox_v2 import CONTEXTS
from .ratio_outpaint import RATIOS
from .resolution_plan import plan, report

RATIO_OPTIONS = ["none"] + list(RATIOS.keys())


class AioliResolutionPanel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AioliResolutionPanel",
            display_name="📏 Aioli Resolution Panel",
            category="Aioli Nodes",
            description="Regle et montre la resolution de la chaine inpaint / outpaint V2 "
                        "(scale, outpaint, crop, taille modele, bandes), avant et apres run.",
            is_output_node=True,
            inputs=[
                io.Image.Input("image", optional=True,
                               tooltip="Image source (LoadImage) : sa taille sert au schema."),
                io.Mask.Input("mask", optional=True,
                              tooltip="Masque dessine (sortie MASK du LoadImage) : regions du schema."),
                io.Float.Input("scale", default=1.0, min=0.01, max=8.0, step=0.01,
                               display_name="SCALE"),
                io.Combo.Input("ratio", options=RATIO_OPTIONS, default="none",
                               display_name="OUTPAINT RATIO"),
                io.Boolean.Input("regenerate_all", default=False,
                                 display_name="REGENERATE WHOLE IMAGE"),
                io.Int.Input("resolution", default=1024, min=0, max=MAX_RESOLUTION, step=64,
                             display_name="RESOLUTION"),
                io.Combo.Input("multiple", options=MULTIPLES, default="16 (Flux)"),
                io.Boolean.Input("force_square", default=False, display_name="SQUARE"),
                io.Boolean.Input("force_target_downscale", default=True,
                                 display_name="DOWNSCALE TO RESOLUTION"),
                io.Int.Input("padding", default=48, min=0, max=4096),
                io.Combo.Input("context", options=CONTEXTS, default="region",
                               display_name="CONTEXT"),
            ],
            outputs=[
                io.Float.Output("scale_out", display_name="SCALE"),
                io.Combo.Output("ratio_out", display_name="OUTPAINT RATIO", options=RATIO_OPTIONS),
                io.Boolean.Output("regenerate_all_out", display_name="REGENERATE WHOLE IMAGE"),
                io.Int.Output("resolution_out", display_name="RESOLUTION"),
                io.Combo.Output("multiple_out", display_name="multiple", options=MULTIPLES),
                io.Boolean.Output("force_square_out", display_name="SQUARE"),
                io.Boolean.Output("force_target_downscale_out", display_name="DOWNSCALE TO RESOLUTION"),
                io.Int.Output("padding_out", display_name="padding"),
                io.Combo.Output("context_out", display_name="CONTEXT", options=CONTEXTS),
                io.String.Output("report", display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, scale, ratio, regenerate_all, resolution, multiple, force_square,
                force_target_downscale, padding, context, image=None, mask=None) -> io.NodeOutput:
        settings = dict(scale=scale, ratio=ratio, regenerate_all=regenerate_all,
                        resolution=resolution, multiple=multiple, force_square=force_square,
                        force_target_downscale=force_target_downscale, padding=padding,
                        context=context)
        text = ""
        ui = None
        if image is not None:
            H, W = int(image.shape[1]), int(image.shape[2])
            p = plan(W, H, mask, **settings)
            text = report(p)
            p["origin"] = "run"
            ui = {"aioli_plan": [p]}
            print("[AioliResolutionPanel]\n" + text)
        return io.NodeOutput(scale, ratio, regenerate_all, resolution, multiple, force_square,
                             force_target_downscale, padding, context, text, ui=ui)
