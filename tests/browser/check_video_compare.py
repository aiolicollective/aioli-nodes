"""
Verification du 🎞️ Aioli Video Compare dans un vrai frontend ComfyUI, en legacy
puis en Nodes 2.0. Hors pytest (ComfyUI lance avec aioli-nodes + Playwright) :

    python tests/browser/check_video_compare.py [--url ...] [--chromium ...] [--shots dossier]

Clips de test ecrits dans input/ : A et B (48 images, 24 fps), et un couple
"interpolation x2" (A 28 images a 24 fps, B 56 images a 48 fps). Le numero
d'image est inscrit en gris dans un carre. Pendant la lecture, le script lit ce
carre a gauche (A) et a droite (B) dans le canvas : a fps egaux A et B montrent
la meme image ; en x2, B montre l'image 2a ou 2a+1. Il verifie aussi le pas a
pas, la pleine resolution sans perte en pause, aucune reponse 404 (pas de
tempete de requetes) et aucune erreur console. Effets de bord : 4 fichiers dans
input/, des runs (sans modele), le reglage Comfy.VueNodes.Enabled bascule puis
revient a sa valeur effective.
"""

import argparse
import io
import json
import os
import sys
import time
import urllib.request
import uuid

import numpy as np

# nom, images, fps ; numero d'image i inscrit en gris 20 + 4 i (robuste a +-1 niveau d'encodage)
CLIPS = {"a": ("aioli_vc_check_a.mp4", 48, 24), "b": ("aioli_vc_check_b.mp4", 48, 24),
         "a28": ("aioli_vc_check_a28.mp4", 28, 24), "x2": ("aioli_vc_check_x2.mp4", 56, 48)}
W, H = 640, 360


def make_clip(variant):
    _, N, FPS = CLIPS[variant]
    import av
    buf = io.BytesIO()
    out = av.open(buf, "w", format="mp4")
    st = out.add_stream("libx264", rate=FPS)
    st.width, st.height, st.pix_fmt = W, H, "yuv420p"
    st.options = {"crf": "12"}
    yy, xx = np.mgrid[0:H, 0:W]
    for i in range(N):
        img = np.zeros((H, W, 3), np.uint8)
        img[..., 0] = (xx + i * 8) % 256
        img[..., 1] = (yy * 2 + i * 4) % 256
        img[..., 2] = 128 if variant in ("a", "a28") else 160
        img[30:110, 30:110, :] = 20 + 4 * i               # numero d'image
        for p in st.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
            out.mux(p)
    for p in st.encode():
        out.mux(p)
    out.close()
    return buf.getvalue()


def api(url, path, data=None, method=None, headers=None):
    req = urllib.request.Request(url + path, data=data, method=method, headers=headers or {})
    return urllib.request.urlopen(req).read()


def upload(url, name, data):
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: video/mp4\r\n\r\n").encode() + data + \
        f"\r\n--{b}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{b}--\r\n".encode()
    api(url, "/api/upload/image", body, "POST", {"Content-Type": f"multipart/form-data; boundary={b}"})


BUILD = """async ([ca, cb]) => {
  const app = window.app, LG = window.LiteGraph;
  app.graph.clear();
  const add = (t, x, y) => { const n = LG.createNode(t); n.pos = [x, y]; app.graph.add(n); return n; };
  const la = add("LoadVideo", 0, 0); la.widgets.find(w => w.name === "file").value = ca;
  const lb = add("LoadVideo", 0, 400); lb.widgets.find(w => w.name === "file").value = cb;
  const vc = add("AioliVideoCompare", 360, 0);
  la.connect(0, vc, vc.findInputSlot("a")); lb.connect(0, vc, vc.findInputSlot("b"));
  await app.queuePrompt(0, 1);
  return vc.id;
}"""

# numeros d'image lus dans le canvas, au centre du carre (source 70,70), mode cote a cote
READ = """(id) => {
  const n = window.app.graph.getNodeById(id), S = n._aioliVC;
  const c = n.widgets.find(w => w.name === "video_compare").element.querySelector("canvas");
  const ctx = c.getContext("2d"), cw = c.width, ch = c.height;
  const [W, H] = S.meta.display, s = Math.min((cw / 2) / W, ch / H);
  const x = (cw / 2 - W * s) / 2, y = (ch - H * s) / 2;
  const g = (px, py) => ctx.getImageData(Math.round(px), Math.round(py), 1, 1).data[1];
  const va = g(x + 70 * s, y + 70 * s), vb = g(cw / 2 + x + 70 * s, y + 70 * s);
  return { a: Math.round((va - 20) / 4), b: Math.round((vb - 20) / 4), ia: S.shown.ia, ib: S.shown.ib,
           info: n.widgets.find(w => w.name === "video_compare").element.querySelector(".aioli-vc-info").textContent };
}"""

CTRL = """([id, what, val]) => {
  const n = window.app.graph.getNodeById(id);
  const root = n.widgets.find(w => w.name === "video_compare").element;
  const buttons = root.querySelectorAll("button"), selects = root.querySelectorAll("select");
  if (what === "click") buttons[val].click();
  if (what === "mode") { selects[0].value = val; selects[0].dispatchEvent(new Event("change")); }
  if (what === "zoom") { selects[1].value = val; selects[1].dispatchEvent(new Event("change")); }
  if (what === "key") root.dispatchEvent(new KeyboardEvent("keydown", { key: val, bubbles: true }));
  return true;
}"""


def main():
    from playwright.sync_api import sync_playwright
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8188")
    ap.add_argument("--chromium", default=os.environ.get("AIOLI_CHROMIUM"))
    ap.add_argument("--shots")
    args = ap.parse_args()
    url = args.url.rstrip("/")
    for v, (name, *_r) in CLIPS.items():
        upload(url, name, make_clip(v))

    def set_vue(v):
        api(url, "/api/settings/Comfy.VueNodes.Enabled", json.dumps(v).encode(), "POST",
            {"Content-Type": "application/json"})

    ok, vue_before = True, None
    try:
        with sync_playwright() as pw:
            launch = {"args": ["--headless=new", "--no-sandbox"]}
            if args.chromium:
                launch["executable_path"] = args.chromium
            browser = pw.chromium.launch(**launch)
            p0 = browser.new_page()
            p0.goto(url)
            p0.wait_for_function("() => window.app && window.app.extensionManager", timeout=60000)
            vue_before = bool(p0.evaluate("() => window.app.extensionManager.setting.get('Comfy.VueNodes.Enabled')"))
            p0.close()
            for mode, vue in (("legacy", False), ("nodes2", True)):
                for case, clip_a, clip_b in (("same fps", "a", "b"), ("interpolation x2", "a28", "x2")):
                    set_vue(vue)
                    page = browser.new_context(viewport={"width": 1400, "height": 950}).new_page()
                    errs, gone = [], []
                    page.on("pageerror", lambda e, errs=errs: errs.append(str(e)))
                    page.on("console", lambda m, errs=errs: errs.append(m.text)
                            if m.type == "error" and "Failed to load resource" not in m.text else None)
                    page.on("response", lambda r, gone=gone: gone.append(r.url)
                            if r.status == 404 and "aioli_vcmp" in r.url else None)   # nos images seulement
                    page.goto(url)
                    page.wait_for_function("() => window.app && window.app.graph", timeout=60000)
                    time.sleep(3)
                    page.keyboard.press("Escape")
                    vid = page.evaluate(BUILD, [CLIPS[clip_a][0], CLIPS[clip_b][0]])
                    page.wait_for_function(f"() => window.app.graph.getNodeById({vid})._aioliVC.meta", timeout=120000)
                    page.evaluate(CTRL, [vid, "mode", "side"])
                    time.sleep(2)
                    page.evaluate(CTRL, [vid, "click", 0])           # lecture
                    samples = []
                    for _ in range(25):
                        time.sleep(0.09)
                        samples.append(page.evaluate(READ, vid))
                    page.evaluate(CTRL, [vid, "click", 0])           # pause
                    time.sleep(2)
                    if clip_b == "x2":
                        desync = [x for x in samples if x["b"] // 2 != x["a"]]
                    else:
                        desync = [x for x in samples if x["a"] != x["b"]]
                    moved = len({x["a"] for x in samples})
                    paused = page.evaluate(READ, vid)
                    full_fit = "full resolution" in paused["info"]   # pause en "fit" : sans perte
                    page.evaluate(CTRL, [vid, "key", "ArrowRight"])
                    time.sleep(1.5)
                    r1 = page.evaluate(READ, vid)
                    stepped = r1["ia"] == paused["ia"] + 1 or (clip_b == "x2" and r1["ib"] == paused["ib"] + 1)
                    if args.shots and case == "same fps":
                        os.makedirs(args.shots, exist_ok=True)
                        for m in ("wipe", "side", "diff"):
                            page.evaluate(CTRL, [vid, "mode", m])
                            time.sleep(1.5)
                            el = page.locator(f'[data-node-id="{vid}"]') if vue else None
                            path = os.path.join(args.shots, f"vc_{mode}_{m}.png")
                            (el.screenshot(path=path) if el and el.count() else page.screenshot(path=path))
                    res = dict(samples=len(samples), distinct_frames=moved, desync=len(desync),
                               step_ok=stepped, full_on_pause=full_fit, http_404=len(gone), console_errors=errs)
                    good = (moved >= 5 and not desync and stepped and full_fit and not gone and not errs)
                    ok &= good
                    print(f"[{mode}] {case}: {'OK' if good else 'ECHEC'} {res}"
                          + (f" desync: {desync[:3]}" if desync else ""))
                    page.context.close()
            browser.close()
    finally:
        if vue_before is not None:
            set_vue(vue_before)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
