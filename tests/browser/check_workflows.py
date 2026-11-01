"""
Verification des wf d'exemple (examples/*.json) dans un vrai frontend ComfyUI,
en legacy puis en Nodes 2.0. Hors pytest (ComfyUI lance avec aioli-nodes +
KJNodes, Playwright) :

    python tests/browser/check_workflows.py [--url ...] [--chromium ...] [--shots dossier]

Pour chaque wf : aucune exception JS ni erreur console (hors ressources :
images et modeles absents), conversion en prompt (graphToPrompt) sans erreur,
widgets des nodes aioli identiques apres enregistrement (un wf plus ancien que
le node recoit en plus, a la fin, les widgets ajoutes depuis). Pour les wf qui ont un
📏 Resolution Panel : ses 9 reglages branches sur le node subgraph, et chaque
entree qu'il alimente accepte son type (regle de comfy_execution/validation.py).
Rien n'est mis en file d'attente. Effet de bord : le reglage
Comfy.VueNodes.Enabled bascule puis revient a sa valeur effective.
"""

import argparse
import glob
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AIOLI = ("RatioOutpaintCalc", "AioliMaskBBoxV2", "InpaintColorFix", "BBoxMultipleAssembler", "RegionMaskList",
         "MaskSplitRegions", "RegionPreview", "RegionalMaskConditioning", "AioliMaskBBox", "BBoxMultipleFix",
         "AioliResolutionPanel", "AioliVideoCompare")

LOAD = """async (wf) => {
  const app = window.app;
  await app.loadGraphData(wf, true, true, null);
  await new Promise(r => setTimeout(r, 2500));
  const out = { prompt_error: null, panel: null };
  let prompt = null;
  try { prompt = await app.graphToPrompt(); } catch (e) { out.prompt_error = String(e); }
  out.saved = JSON.stringify(app.rootGraph.serialize());
  const p = app.rootGraph.nodes.find(n => n.type === "AioliResolutionPanel");
  const sg = app.rootGraph.nodes.find(n => n.isSubgraphNode?.());
  if (p && sg) {
    const linked = sg.inputs.filter(i => i.link != null && app.rootGraph.links.get?.(i.link)?.origin_id == p.id);
    const uses = [];
    if (prompt) for (const n of Object.values(prompt.output))
      for (const [inp, v] of Object.entries(n.inputs || {}))
        if (Array.isArray(v) && String(v[0]) === String(p.id)) uses.push([n.class_type, inp, v[1]]);
    const ml = sg.widgets?.find(w => w.name === "masked_latent");
    out.panel = { linked: linked.length, uses, masked_latent: ml ? ml.value : null };
  }
  return out;
}"""


def api(url, path, data=None, method=None, headers=None):
    req = urllib.request.Request(url + path, data=data, method=method, headers=headers or {})
    return urllib.request.urlopen(req).read()


def aioli_widgets(wf):
    out = {}
    for g in [wf] + wf.get("definitions", {}).get("subgraphs", []):
        for n in g.get("nodes", []):
            if n.get("type") in AIOLI:
                out[f"{n['type']}#{n['id']}"] = n.get("widgets_values")
    return out


def type_errors(info, uses):
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
            errs.append(f"{cls}.{inp}: {got} -> {want if isinstance(want, str) else 'list'}")
    return errs


def main():
    from playwright.sync_api import sync_playwright
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8188")
    ap.add_argument("--chromium", default=os.environ.get("AIOLI_CHROMIUM"))
    ap.add_argument("--shots")
    args = ap.parse_args()
    url = args.url.rstrip("/")
    info = json.loads(api(url, "/api/object_info"))
    wfs = sorted(glob.glob(os.path.join(ROOT, "examples", "*.json")))

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
                set_vue(vue)
                page = browser.new_context(viewport={"width": 1600, "height": 1000}).new_page()
                errs = []
                page.on("pageerror", lambda e: errs.append(str(e)))
                page.on("console", lambda m: errs.append(m.text)
                        if m.type == "error" and "Failed to load resource" not in m.text else None)
                page.goto(url)
                page.wait_for_function("() => window.app && window.app.graph", timeout=60000)
                time.sleep(3)
                page.keyboard.press("Escape")
                for path in wfs:
                    errs.clear()
                    wf = json.load(open(path, encoding="utf-8"))
                    res = page.evaluate(LOAD, wf)
                    name = os.path.basename(path)
                    before, after = aioli_widgets(wf), aioli_widgets(json.loads(res["saved"]))
                    # un wf plus ancien que le node recoit les widgets ajoutes depuis, a la fin
                    # (valeurs par defaut) : seul ce qui existait doit etre identique
                    diffs = [k for k in before if before[k] != (after.get(k) or [])[:len(before[k] or [])]]
                    problems = []
                    if errs:
                        problems.append(f"console: {errs[:3]}")
                    if res["prompt_error"]:
                        problems.append(f"graphToPrompt: {res['prompt_error']}")
                    if diffs:
                        problems.append(f"widgets aioli changes: {diffs[:4]}")
                    p = res["panel"]
                    if p is not None:
                        if p["linked"] != 9:
                            problems.append(f"panneau : {p['linked']} reglages branches sur 9")
                        te = type_errors(info, p["uses"])
                        if te:
                            problems.append(f"types : {te}")
                        if p["masked_latent"] not in (None, False):
                            problems.append("MASKED LATENT n'est pas off par defaut")
                    if args.shots:
                        os.makedirs(args.shots, exist_ok=True)
                        page.screenshot(path=os.path.join(args.shots, f"{mode}_{name[:-5]}.png"))
                    ok &= not problems
                    extra = f" panneau: {p['linked']} liens, {len(p['uses'])} entrees alimentees" if p else ""
                    print(f"[{mode}] {name}: {'OK' if not problems else 'ECHEC ' + '; '.join(problems)}{extra}")
                page.context.close()
            browser.close()
    finally:
        if vue_before is not None:
            set_vue(vue_before)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
