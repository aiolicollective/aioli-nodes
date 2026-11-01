"""
Verification du 📏 Resolution Panel dans un vrai frontend ComfyUI, en legacy
puis en Nodes 2.0 : apercu live, execution reelle de la chaine pilotee par le
panneau, valeurs du run = apercu, aucune erreur console.

A relancer a chaque nouvelle version de ComfyUI / du frontend (Desktop).
Pas un test pytest (il faut un ComfyUI lance, avec aioli-nodes, et Playwright) :

    python tests/browser/check_panel.py [--url http://127.0.0.1:8188] [--chromium /chemin/chrome]
                                        [--workflow wf.json] [--shots dossier]

--workflow : un wf V2 avec un node subgraph ; le panneau est branche sur ses
entrees, le prompt est construit par le frontend (graphToPrompt) mais JAMAIS
envoye : chaque entree alimentee par le panneau est verifiee contre son type
dans /object_info, avec la regle du serveur (une sortie "*" passe partout, une
sortie "COMBO" vers une liste d'options est refusee).

Effets de bord (outil manuel) : ecrit aioli_panel_check.png dans input/ (ecrase),
lance deux petits runs de la chaine aioli (sans modele), bascule le reglage
Comfy.VueNodes.Enabled et le remet a sa valeur effective du debut.
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
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from chain import make_image, make_mask  # noqa: E402

IMAGE_NAME = "aioli_panel_check.png"

BUILD = """async (img) => {
  const app = window.app, LG = window.LiteGraph;
  app.graph.clear();
  const add = (t, x, y) => { const n = LG.createNode(t); n.pos = [x, y]; app.graph.add(n); return n; };
  const W = (n, k, v) => { n.widgets.find(w => w.name === k).value = v; };
  const C = (a, o, b, i) => {
    const oo = typeof o === "string" ? a.findOutputSlot(o) : o;
    const ii = typeof i === "string" ? b.findInputSlot(i) : i;
    if (oo < 0 || ii < 0) throw new Error(`slot ${a.type}.${o}=${oo} -> ${b.type}.${i}=${ii}`);
    if (!a.connect(oo, b, ii)) throw new Error(`lien refuse ${a.type}.${o} -> ${b.type}.${i}`);
  };
  const li = add("LoadImage", 0, 0); W(li, "image", img);
  const p = add("AioliResolutionPanel", 350, 0);
  C(li, 0, p, "image"); C(li, 1, p, "mask");
  const sc = add("ImageScaleBy", 900, 0); W(sc, "upscale_method", "lanczos");
  C(li, 0, sc, "image"); C(p, "OUTPUT SCALE", sc, "scale_by");
  const ro = add("RatioOutpaintCalc", 1150, 0); W(ro, "multiple", "none");
  C(sc, 0, ro, "image"); C(li, 1, ro, "mask"); C(p, "OUTPAINT RATIO", ro, "ratio");
  const ms = add("MaskSplitRegions", 1150, 300); C(ro, 2, ms, "mask");
  const rl = add("RegionMaskList", 1400, 300); W(rl, "add_background", false); W(rl, "empty", "full_image");
  C(ms, 0, rl, "masks"); C(ro, 1, rl, "extra_mask");
  const bb = add("AioliMaskBBoxV2", 1650, 0);
  C(ro, 0, bb, "image"); C(rl, 0, bb, "mask");
  for (const [o, i] of [["padding", "padding"], ["multiple", "multiple"], ["MODEL SIZE", "resolution"],
                        ["SQUARE", "force_square"], ["SHRINK BIG CROPS", "force_target_downscale"],
                        ["CONTEXT", "context"]]) C(p, o, bb, i);
  const as = add("BBoxMultipleAssembler", 1950, 0);
  C(ro, 0, as, "base_image"); C(bb, 0, as, "crops"); C(bb, 1, as, "masks");
  C(bb, 2, as, "x"); C(bb, 3, as, "y"); C(bb, 4, as, "width"); C(bb, 5, as, "height");
  const pv = add("PreviewImage", 2250, 0); C(as, 0, pv, "images");
  for (const [k, v] of [["scale", 1.5], ["ratio", "16:9"], ["resolution", 1024], ["force_square", false]]) {
    const w = p.widgets.find(w => w.name === k); w.value = v; w.callback?.(v);
  }
  return { panel: p.id, preview: pv.id };
}"""

SUBGRAPH = """async () => {
  const app = window.app, LG = window.LiteGraph;
  const sg = app.graph.nodes.find(n => n.isSubgraphNode?.());
  if (!sg) return { error: "pas de node subgraph dans ce wf" };
  const p = LG.createNode("AioliResolutionPanel"); p.pos = [-3000, -1300]; app.graph.add(p);
  const linked = {};
  for (const [o, i] of [["OUTPUT SCALE", "scale"], ["OUTPAINT RATIO", "ratio"], ["REGENERATE WHOLE IMAGE", "regenerate_all"],
                        ["MODEL SIZE", "resolution"], ["multiple", "multiple"], ["SQUARE", "force_square"],
                        ["SHRINK BIG CROPS", "force_target_downscale"], ["padding", "padding"], ["CONTEXT", "context"]]) {
    const oo = p.findOutputSlot(o), ii = sg.findInputSlot(i);
    linked[i] = oo >= 0 && ii >= 0 ? !!p.connect(oo, sg, ii) : "absent";
  }
  const prompt = await app.graphToPrompt();            // jamais envoye au serveur
  const uses = [];
  for (const [id, n] of Object.entries(prompt.output))
    for (const [inp, v] of Object.entries(n.inputs || {}))
      if (Array.isArray(v) && String(v[0]) === String(p.id)) uses.push([n.class_type, inp, v[1]]);
  return { linked, uses };
}"""


def type_errors(url, uses):
    """
    Regle de validate_node_input (comfy_execution/validation.py) pour chaque
    entree alimentee par le panneau : type de sortie du panneau vs type attendu.
    """
    info = json.loads(api(url, "/api/object_info"))
    out_types = info["AioliResolutionPanel"]["output"]
    errs = []
    for cls, inp, slot in uses:
        spec = info.get(cls, {}).get("input", {})
        want = (spec.get("required", {}).get(inp) or spec.get("optional", {}).get(inp) or [None])[0]
        got = out_types[slot]
        ok = (got == want or got == "*" or want == "*"
              or (isinstance(want, str) and isinstance(got, str)
                  and set(t.strip() for t in got.split(",")) & set(t.strip() for t in want.split(","))))
        if not ok:
            errs.append(f"{cls}.{inp}: {got} -> {want if isinstance(want, str) else 'liste'}")
    return errs


def api(url, path, data=None, method=None, headers=None):
    req = urllib.request.Request(url + path, data=data, method=method, headers=headers or {})
    return urllib.request.urlopen(req).read()


def upload_image(url):
    """Image de test (1000x667, 3 taches dessinees en alpha) envoyee dans input/."""
    src = make_image(1000, 667, seed=21)
    mask = make_mask(1000, 667, "multi", seed=21)
    rgb = (src[0].numpy() * 255).round().astype(np.uint8)
    alpha = ((1 - mask[0].numpy()) * 255).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(np.dstack([rgb, alpha]), "RGBA").save(buf, format="PNG")
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{IMAGE_NAME}\"\r\n"
            f"Content-Type: image/png\r\n\r\n").encode() + buf.getvalue() + \
           f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n".encode()
    api(url, "/api/upload/image", body, "POST", {"Content-Type": f"multipart/form-data; boundary={boundary}"})


def main():
    from playwright.sync_api import sync_playwright

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8188")
    ap.add_argument("--chromium", default=os.environ.get("AIOLI_CHROMIUM"))
    ap.add_argument("--workflow")
    ap.add_argument("--shots")
    args = ap.parse_args()
    url = args.url.rstrip("/")

    upload_image(url)
    vue_before = None

    def set_vue(v):
        api(url, "/api/settings/Comfy.VueNodes.Enabled", json.dumps(v).encode(), "POST",
            {"Content-Type": "application/json"})

    ok = True
    try:
        with sync_playwright() as pw:
            launch = {"args": ["--headless=new", "--no-sandbox"]}
            if args.chromium:
                launch["executable_path"] = args.chromium
            browser = pw.chromium.launch(**launch)
            # valeur effective du reglage (defaut compris), pour la remettre a la fin
            page0 = browser.new_page()
            page0.goto(url)
            page0.wait_for_function("() => window.app && window.app.extensionManager", timeout=60000)
            vue_before = bool(page0.evaluate(
                "() => window.app.extensionManager.setting.get('Comfy.VueNodes.Enabled')"))
            page0.close()
            for mode, vue in (("legacy", False), ("nodes2", True)):
                set_vue(vue)
                page = browser.new_context(viewport={"width": 1500, "height": 950}).new_page()
                errs = []
                page.on("pageerror", lambda e, errs=errs: errs.append(str(e)))
                page.on("console", lambda m, errs=errs: errs.append(m.text)
                        if m.type == "error" and "Failed to load resource" not in m.text else None)
                page.goto(url)
                page.wait_for_function("() => window.app && window.app.graph", timeout=60000)
                time.sleep(3)
                page.keyboard.press("Escape")
                ids = page.evaluate(BUILD, IMAGE_NAME)
                time.sleep(3)
                text = "(id) => window.app.graph.getNodeById(id)._aioliRP.text.innerText"
                preview = page.evaluate(text, ids["panel"])
                before = set(json.loads(api(url, "/api/history?max_items=500")))
                page.evaluate("async () => { await window.app.queuePrompt(0, 1); }")
                run = {}
                for _ in range(300):
                    hist = json.loads(api(url, "/api/history?max_items=500"))
                    new = [v for k, v in hist.items() if k not in before]
                    if new and new[0].get("status", {}).get("completed") is not None:
                        run = new[0]
                        break
                    time.sleep(1)
                time.sleep(1)
                after = page.evaluate(text, ids["panel"])
                if args.shots:
                    os.makedirs(args.shots, exist_ok=True)
                    page.screenshot(path=os.path.join(args.shots, f"panel_{mode}.png"))
                res = {
                    "status": run.get("status", {}).get("status_str"),
                    "image": bool(run.get("outputs", {}).get(str(ids["preview"]), {}).get("images")),
                    "preview_ok": bool(preview.strip()),
                    "run_equals_preview": preview.replace("preview before run", "") ==
                                          after.replace("values of the last run", ""),
                    "console_errors": errs,
                }
                if args.workflow:
                    page.evaluate("async (wf) => { await window.app.loadGraphData(wf, true, true, 'check'); }",
                                  json.load(open(args.workflow)))
                    time.sleep(4)
                    page.keyboard.press("Escape")
                    sg = page.evaluate(SUBGRAPH)
                    res["subgraph_links"] = sg.get("linked")
                    res["type_errors"] = type_errors(url, sg.get("uses", []))
                    res["inputs_checked"] = len(sg.get("uses", []))
                good = (res["status"] == "success" and res["image"] and res["preview_ok"]
                        and res["run_equals_preview"] and not errs and not res.get("type_errors"))
                ok &= good
                print(f"[{mode}] {'OK' if good else 'ECHEC'} {res}")
            browser.close()
    finally:
        if vue_before is not None:
            set_vue(vue_before)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
