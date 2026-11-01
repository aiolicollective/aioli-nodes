"""
🎞️ Aioli Video Compare (cle "AioliVideoCompare", API V3).

Compare deux clips (IMAGE batch ou VIDEO) a l'image pres : wipe, cote a cote,
difference, bascule A / B, inversion ; lecture, scrub, pas a pas, zoom sur des
images pleine resolution sans perte. Widget : web/aioli_video_compare.js.

- Rien n'est garde en memoire : les images sont ecrites une fois (PNG sans perte
  + proxy JPEG) dans le temp de ComfyUI et lues par /view.
- Une session par node : elle n'est remplacee que quand CE node est relance
  (un node en cache garde une session valide). Au plus MAX_NODES nodes ; le
  dossier temp est vide par ComfyUI a chaque demarrage.
- Geometrie d'affichage : la plus grande des deux ; l'autre clip est agrandi a
  l'affichage (ses pixels ne sont pas modifies), et c'est signale.
- Un fps par clip (IMAGE : fps_a / fps_b ; VIDEO : son propre fps). Le plus
  court reste sur sa derniere image.
"""

import os
import shutil
import threading
import uuid
from collections import OrderedDict

import folder_paths
from comfy_api.latest import io

from .video_compare_core import display_geometry, iter_frames, proxy_size, same_clock, write_clip

SUBFOLDER = "aioli_vcmp"
MAX_NODES = 16
_SESSIONS = OrderedDict()      # unique_id du node -> dossier de sa derniere session
_LOCK = threading.Lock()


def _fps_input(key, label):
    return io.Float.Input(key, default=24.0, min=0.1, max=240.0, step=0.001, display_name=label,
                          tooltip="Frame rate when this side is an IMAGE batch (a VIDEO keeps its own).")


class AioliVideoCompare(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AioliVideoCompare",
            display_name="🎞️ Aioli Video Compare",
            category="Aioli Nodes",
            description="Compare two clips frame by frame: wipe, side by side, difference, "
                        "toggle, swap; play, scrub, step, zoom on lossless full-resolution frames.",
            is_output_node=True,
            inputs=[
                io.MultiType.Input("a", types=[io.Image, io.Video], display_name="A"),
                io.MultiType.Input("b", types=[io.Image, io.Video], display_name="B"),
                _fps_input("fps_a", "fps A (IMAGE)"),
                _fps_input("fps_b", "fps B (IMAGE)"),
                io.String.Input("label_a", default="A"),
                io.String.Input("label_b", default="B"),
            ],
            outputs=[],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, a, b, fps_a, fps_b, label_a, label_b) -> io.NodeOutput:
        sid = uuid.uuid4().hex[:12]
        folder = os.path.join(folder_paths.get_temp_directory(), SUBFOLDER, sid)
        meta = {}
        try:
            for side, value, fps, label in (("a", a, fps_a, label_a), ("b", b, fps_b, label_b)):
                rate, frames = iter_frames(value, fps)
                n, w, h = write_clip(frames, folder, side)
                meta[side] = dict(label=str(label), n=n, fps=rate, w=w, h=h, duration=n / rate)
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        ma, mb = meta["a"], meta["b"]
        display = display_geometry(ma, mb)
        warnings = []
        for k, m in (("A", ma), ("B", mb)):
            if (m["w"], m["h"]) != display:
                warnings.append(f"{k} {m['w']}x{m['h']} is shown enlarged to {display[0]}x{display[1]} "
                                f"(its pixels are not changed)")
        if ma["w"] * mb["h"] != mb["w"] * ma["h"]:
            warnings.append("A and B have different aspect ratios: each is fitted, not stretched")
        clock = "common" if same_clock(ma["fps"], mb["fps"]) else "separate"
        if clock == "separate":
            warnings.append(f"different frame rates: A {ma['fps']:.4f} fps, B {mb['fps']:.4f} fps")
        if ma["n"] != mb["n"]:
            warnings.append(f"different lengths: A {ma['n']} frames, B {mb['n']} frames "
                            f"(the shorter one holds its last frame)")

        # une session par node : on remplace la sienne ; suppression hors verrou
        uid = str(cls.hidden.unique_id) if cls.hidden and cls.hidden.unique_id else sid
        stale = []
        with _LOCK:
            old = _SESSIONS.pop(uid, None)
            if old:
                stale.append(old)
            _SESSIONS[uid] = folder
            while len(_SESSIONS) > MAX_NODES:
                stale.append(_SESSIONS.popitem(last=False)[1])
        for f in stale:
            shutil.rmtree(f, ignore_errors=True)

        out = dict(sid=sid, subfolder=f"{SUBFOLDER}/{sid}", a=ma, b=mb, display=list(display),
                   proxy=list(proxy_size(*display)), clock=clock, warnings=warnings)
        print(f"[AioliVideoCompare] A {ma['n']} x {ma['w']}x{ma['h']} @ {ma['fps']:g} fps | "
              f"B {mb['n']} x {mb['w']}x{mb['h']} @ {mb['fps']:g} fps | {folder}")
        return io.NodeOutput(ui={"aioli_vcompare": [out]})
