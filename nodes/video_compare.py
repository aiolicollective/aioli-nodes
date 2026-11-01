"""
🎞️ Aioli Video Compare (cle "AioliVideoCompare", API V3).

Compare deux clips (IMAGE batch ou VIDEO) a l'image pres : wipe, cote a cote,
difference, bascule A / B, inversion ; lecture, scrub, pas a pas, zoom sur des
images pleine resolution sans perte. Widget : web/aioli_video_compare.js.

- Rien n'est garde en memoire : les images sont ecrites une fois (PNG sans perte
  + proxy JPEG) dans le temp de ComfyUI et lues par /view.
- Le node s'execute a chaque run (not_idempotent) : son widget pointe toujours
  vers des images qui existent, quel que soit le cache de ComfyUI. Si ses
  entrees ont le meme contenu (empreinte exacte des images, ou meme fichier
  video inchange), il reutilise les images deja ecrites. Il garde ses
  KEEP_PER_NODE dernieres sessions (allers-retours A / B entre reglages) ; au
  plus MAX_NODES nodes ; le temp est vide par ComfyUI a chaque demarrage.
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
import torch
from comfy_api.latest import io

from .video_compare_core import (display_geometry, iter_frames, proxy_size, same_clock, same_input,
                                 signature, write_clip)

SUBFOLDER = "aioli_vcmp"
MAX_NODES = 16
KEEP_PER_NODE = 3
_NODES = OrderedDict()         # unique_id -> liste de sessions [{sid, folder, sig, clips}]
_LOCK = threading.Lock()


def _fps_input(key, label):
    return io.Float.Input(key, default=24.0, min=0.1, max=240.0, step=0.001, display_name=label,
                          tooltip="Frame rate when this side is an IMAGE batch (a VIDEO keeps its own).")


def _find(uid, a, b, sig_a, sig_b):
    """Session existante de ce node pour ces memes entrees (dossier encore present)."""
    with _LOCK:
        sessions = list(_NODES.get(uid, []))
    for sess in sessions:
        if (same_input(sess["sig"][0], a, sig_a) and same_input(sess["sig"][1], b, sig_b)
                and os.path.isdir(sess["folder"])):
            return sess
    return None


def _remember(uid, sess):
    stale = []
    with _LOCK:
        lst = [x for x in _NODES.pop(uid, []) if x["sid"] != sess["sid"]]
        lst.insert(0, sess)
        stale += [x["folder"] for x in lst[KEEP_PER_NODE:]]
        _NODES[uid] = lst[:KEEP_PER_NODE]
        while len(_NODES) > MAX_NODES:
            stale += [x["folder"] for x in _NODES.popitem(last=False)[1]]
    for f in stale:                      # suppression hors verrou
        shutil.rmtree(f, ignore_errors=True)


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
            not_idempotent=True,
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
        uid = str(cls.hidden.unique_id) if cls.hidden and cls.hidden.unique_id else "?"
        sig_a, sig_b = signature(a), signature(b)
        sess = _find(uid, a, b, sig_a, sig_b)
        reused = sess is not None
        if not reused:
            sid = uuid.uuid4().hex[:12]
            folder = os.path.join(folder_paths.get_temp_directory(), SUBFOLDER, sid)
            clips = {}
            try:
                for side, value, fps in (("a", a, fps_a), ("b", b, fps_b)):
                    rate, frames = iter_frames(value, fps)
                    n, w, h = write_clip(frames, folder, side)
                    clips[side] = dict(n=n, fps=rate, w=w, h=h)
            except Exception:
                shutil.rmtree(folder, ignore_errors=True)
                raise
            sess = dict(sid=sid, folder=folder, sig=(sig_a, sig_b), clips=clips,
                        is_image={"a": isinstance(a, torch.Tensor), "b": isinstance(b, torch.Tensor)})
        _remember(uid, sess)

        ma = dict(sess["clips"]["a"], label=str(label_a))
        mb = dict(sess["clips"]["b"], label=str(label_b))
        # fps d'une IMAGE : celui du widget de CE run (changer le fps ne reecrit rien)
        if sess["is_image"]["a"]:
            ma["fps"] = float(fps_a)
        if sess["is_image"]["b"]:
            mb["fps"] = float(fps_b)
        for m in (ma, mb):
            m["duration"] = m["n"] / m["fps"]
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

        out = dict(sid=sess["sid"], subfolder=f"{SUBFOLDER}/{sess['sid']}", a=ma, b=mb,
                   display=list(display), proxy=list(proxy_size(*display)), clock=clock,
                   warnings=warnings)
        print(f"[AioliVideoCompare] {'reused' if reused else 'written'}: A {ma['n']} x {ma['w']}x{ma['h']} "
              f"@ {ma['fps']:g} fps | B {mb['n']} x {mb['w']}x{mb['h']} @ {mb['fps']:g} fps")
        return io.NodeOutput(ui={"aioli_vcompare": [out]})
