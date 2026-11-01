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
from .resolution_plan import RATIO_OPTIONS, plan, report


class AioliResolutionPanel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AioliResolutionPanel",
            display_name="📏 Aioli Resolution Panel",
            category="Aioli Nodes",
            description="Sets and shows how the inpaint / outpaint chain sizes the image "
                        "(output scale, outpaint, crops, model size, bands), live before a run "
                        "and exact after it.",
            is_output_node=True,
            inputs=[
                io.Image.Input("image", optional=True,
                               tooltip="Image source (LoadImage) : sa taille sert au schema."),
                io.Mask.Input("mask", optional=True,
                              tooltip="Masque dessine (sortie MASK du LoadImage) : regions du schema."),
                io.Float.Input("scale", default=1.0, min=0.01, max=8.0, step=0.01,
                               display_name="OUTPUT SCALE",
                               tooltip="Size of the final image = source x this factor. "
                                       "Does not change how much the model works."),
                io.Combo.Input("ratio", options=RATIO_OPTIONS, default="none",
                               display_name="OUTPAINT RATIO"),
                io.Boolean.Input("regenerate_all", default=False,
                                 display_name="REGENERATE WHOLE IMAGE"),
                io.Int.Input("resolution", default=1024, min=0, max=MAX_RESOLUTION, step=64,
                             display_name="MODEL SIZE",
                             tooltip="Model working size per region, square equivalent: 1024 = as many "
                                     "pixels as 1024x1024 (~1 MP), whatever the ratio. Smaller crops are "
                                     "enlarged to it. 0 = native size."),
                io.Combo.Input("multiple", options=MULTIPLES, default="16 (Flux)"),
                io.Boolean.Input("force_square", default=False, display_name="SQUARE"),
                io.Boolean.Input("force_target_downscale", default=True,
                                 display_name="SHRINK BIG CROPS",
                                 tooltip="On: a crop larger than MODEL SIZE^2 is shrunk to it. Off: it keeps "
                                         "its native size, capped at 2048^2 (4.2 MP)."),
                io.Int.Input("padding", default=48, min=0, max=4096,
                             tooltip="Margin around each region's bbox (px). Keep it >= mask_adjust + "
                                     "feather of the assembler, or the fade is cut at the crop edge."),
                io.Combo.Input("context", options=CONTEXTS, default="region",
                               display_name="CONTEXT"),
            ],
            outputs=[
                io.Float.Output("scale_out", display_name="OUTPUT SCALE"),
                # Listes deroulantes en sortie : type "*" (AnyType). Le serveur
                # refuse une sortie "COMBO" vers l'entree liste d'un node V1
                # (comfy_execution/validation.py) ; "*" est accepte partout.
                io.AnyType.Output("ratio_out", display_name="OUTPAINT RATIO"),
                io.Boolean.Output("regenerate_all_out", display_name="REGENERATE WHOLE IMAGE"),
                io.Int.Output("resolution_out", display_name="MODEL SIZE"),
                io.AnyType.Output("multiple_out", display_name="multiple"),
                io.Boolean.Output("force_square_out", display_name="SQUARE"),
                io.Boolean.Output("force_target_downscale_out", display_name="SHRINK BIG CROPS"),
                io.Int.Output("padding_out", display_name="padding"),
                io.AnyType.Output("context_out", display_name="CONTEXT"),
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
