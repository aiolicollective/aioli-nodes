"""
🎞️ Aioli Video Compare (cle "AioliVideoCompare", API V3).

Compare deux clips (IMAGE batch ou VIDEO) cote a cote, a l'image pres : wipe,
cote a cote, difference, bascule A / B, inversion ; lecture, scrub, pas a pas,
zoom 1:1 en pleine resolution sans perte. Widget : web/aioli_video_compare.js.

Les deux clips sont affiches dans la geometrie de A : si B a une autre taille,
il est redimensionne (Lanczos) a la taille de A pour l'affichage, et le
panneau le signale. Longueurs / fps differents : une horloge commune, chaque
clip a son fps, le plus court reste sur sa derniere image.
"""

import os
import shutil
import threading
import uuid
from collections import OrderedDict

from aiohttp import web

import folder_paths
from comfy_api.latest import io
from server import PromptServer

from .video_compare_core import describe, frames_of, full_png, proxy_size, write_proxies

SUBFOLDER = "aioli_vcmp"
_SESSIONS = OrderedDict()      # sid -> {"a": frames, "b": frames, "size": (w, h) d'affichage}
_SESSIONS_MAX = 4
_LOCK = threading.Lock()


class AioliVideoCompare(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AioliVideoCompare",
            display_name="🎞️ Aioli Video Compare",
            category="Aioli Nodes",
            description="Compare two clips frame by frame: wipe, side by side, difference, "
                        "toggle, swap; play, scrub, step, 1:1 zoom on lossless full-resolution frames.",
            is_output_node=True,
            inputs=[
                io.MultiType.Input("a", types=[io.Image, io.Video], display_name="A"),
                io.MultiType.Input("b", types=[io.Image, io.Video], display_name="B"),
                io.Float.Input("fps", default=24.0, min=0.1, max=240.0, step=0.001,
                               tooltip="Frame rate of IMAGE batches (a VIDEO input keeps its own)."),
                io.String.Input("label_a", default="A"),
                io.String.Input("label_b", default="B"),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, a, b, fps, label_a, label_b) -> io.NodeOutput:
        fa, fps_a = frames_of(a, fps)
        fb, fps_b = frames_of(b, fps)
        ma, mb = describe(fa, fps_a, label_a), describe(fb, fps_b, label_b)
        size = (ma["w"], ma["h"])                      # geometrie d'affichage = celle de A
        pw, ph = proxy_size(*size)
        sid = uuid.uuid4().hex[:12]
        sub = f"{SUBFOLDER}/{sid}"
        folder = os.path.join(folder_paths.get_temp_directory(), SUBFOLDER, sid)
        write_proxies(fa, folder, "a", (pw, ph))
        write_proxies(fb, folder, "b", (pw, ph))
        warnings = []
        if (mb["w"], mb["h"]) != size:
            warnings.append(f"B {mb['w']}x{mb['h']} shown at A's size {size[0]}x{size[1]}")
        if abs(fps_a - fps_b) > 1e-6:
            warnings.append(f"different frame rates: A {fps_a:g} fps, B {fps_b:g} fps")
        if ma["n"] != mb["n"]:
            warnings.append(f"different lengths: A {ma['n']} frames, B {mb['n']} frames "
                            f"(the shorter one holds its last frame)")
        with _LOCK:
            _SESSIONS[sid] = {"a": fa, "b": fb, "size": size, "folder": folder}
            while len(_SESSIONS) > _SESSIONS_MAX:
                _, old = _SESSIONS.popitem(last=False)
                shutil.rmtree(old["folder"], ignore_errors=True)
        meta = dict(sid=sid, subfolder=sub, a=ma, b=mb, display=list(size), proxy=[pw, ph],
                    warnings=warnings)
        print(f"[AioliVideoCompare] A {ma['n']} x {ma['w']}x{ma['h']} @ {fps_a:g} fps | "
              f"B {mb['n']} x {mb['w']}x{mb['h']} @ {fps_b:g} fps | proxys {pw}x{ph}")
        return io.NodeOutput(ui={"aioli_vcompare": [meta]})


@PromptServer.instance.routes.get("/aioli/video_compare/frame")
async def video_compare_frame(request):
    """Image pleine resolution sans perte (PNG) : ?sid=...&side=a|b&i=N."""
    q = request.rel_url.query
    sid, side = q.get("sid", ""), q.get("side", "")
    try:
        i = int(q.get("i", ""))
    except ValueError:
        return web.json_response({"error": "i: integer expected"}, status=400)
    with _LOCK:
        sess = _SESSIONS.get(sid)
    if sess is None or side not in ("a", "b"):
        return web.json_response({"error": "unknown session (run the node again)"}, status=404)
    frames = sess[side]
    if not 0 <= i < int(frames.shape[0]):
        return web.json_response({"error": "frame out of range"}, status=404)
    size = sess["size"] if side == "b" else None      # B dans la geometrie de A
    import asyncio
    data = await asyncio.get_running_loop().run_in_executor(None, full_png, frames, i, size)
    return web.Response(body=data, content_type="image/png",
                        headers={"Cache-Control": "private, max-age=3600"})
