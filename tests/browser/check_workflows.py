"""
Verification des wf d'exemple (examples/*.json) dans un vrai frontend ComfyUI,
en legacy puis en Nodes 2.0. Hors pytest (ComfyUI lance avec aioli-nodes +
KJNodes, Playwright) :

    python tests/browser/check_workflows.py [--url ...] [--chromium ...] [--shots dossier] [--dir dossier]

Pour chaque wf : aucune exception JS, aucune erreur console (hors ressources :
images et modeles absents) ni reponse d'erreur des routes /aioli/, conversion en
prompt (graphToPrompt) sans erreur, aucune entree requise absente (nodes de
sortie et ce dont ils dependent, comme le serveur), widgets des
nodes aioli identiques apres enregistrement (un wf plus ancien que le node recoit
en plus, a la fin, les widgets ajoutes depuis).
Wf actuels (ceux qui ont un 📏 Resolution Panel) : tous leurs nodes existent
(aioli-nodes + KJNodes + coeur), le panneau lit image / mask directement sur
les sorties 0 / 1 d'un LoadImage (comme dans les exemples livres) et
envoie chaque sortie sur l'entree du subgraph du meme nom, chaque entree qu'il
alimente accepte son type (regle de comfy_execution/validation.py), et
l'interrupteur MASKED LATENT, quand le wf en a un, existe et est off.
Anciens wf : les nodes d'autres packs absents ici sont seulement comptes.
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
  const out = { prompt_error: null, panel: null, prompt: null };
  let prompt = null;
  try { prompt = await app.graphToPrompt(); out.prompt = prompt.output; } catch (e) { out.prompt_error = String(e); }
  out.saved = JSON.stringify(app.rootGraph.serialize());
  const g = app.rootGraph;
  const p = g.nodes.find(n => n.type === "AioliResolutionPanel");
  const sg = g.nodes.find(n => n.isSubgraphNode?.());
  if (p && sg) {
    const link = (id) => (g.links.get ? g.links.get(id) : g.links[id]);
    const pairs = sg.inputs.filter(i => i.link != null && String(link(i.link)?.origin_id) === String(p.id))
                           .map(i => [link(i.link).origin_slot, i.name]);
    const src = ["image", "mask"].map(nm => { const i = p.inputs.find(i => i.name === nm);
      const l = i && i.link != null ? link(i.link) : null; return l ? [g.getNodeById(l.origin_id)?.type, l.origin_slot] : null; });
    const uses = [];
    if (prompt) for (const n of Object.values(prompt.output))
      for (const [inp, v] of Object.entries(n.inputs || {}))
        if (Array.isArray(v) && String(v[0]) === String(p.id)) uses.push([n.class_type, inp, v[1]]);
    const ml = sg.widgets?.find(w => w.name === "masked_latent");
    out.panel = { pairs, src, uses, masked_widget: ml ? ml.value : null,
                  masked_branch: sg.subgraph.nodes.some(n => n.type === "SetLatentNoiseMask") };
  }
  return out;
}"""

# sortie k du panneau -> entree du subgraph attendue (MODEL SIZE : "crop_resolution" dans le wf Nano Banana)
PANEL_TO = [("scale",), ("ratio",), ("regenerate_all",), ("resolution", "crop_resolution"), ("multiple",),
            ("force_square",), ("force_target_downscale",), ("padding",), ("context",)]


def missing_required(info, prompt):
    """
    Entrees requises absentes, comme le serveur (execution.py) : seulement les
    nodes de sortie et ceux dont ils dependent. Types V3 dynamiques ignores.
    """
    def is_link(v):
        return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)
    stack = [i for i, n in prompt.items() if info.get(n["class_type"], {}).get("output_node")]
    used = set()
    while stack:
        i = stack.pop()
        if i in used or i not in prompt:
            continue
        used.add(i)
        stack += [v[0] for v in prompt[i]["inputs"].values() if is_link(v)]
    out = []
    for nid in sorted(used):
        n = prompt[nid]
        spec = info.get(n["class_type"])
        if not spec:
            continue
        for name, decl in spec.get("input", {}).get("required", {}).items():
            t = decl[0] if isinstance(decl, list) and decl else None
            if isinstance(t, str) and t.startswith("COMFY_") and t.endswith("_V3"):
                continue                     # Autogrow / DynamicCombo : entrees "nom.xxx"
            if name not in n["inputs"]:
                out.append(f"{n['class_type']}#{nid}.{name}")
    return out


FRONT_ONLY = {"MarkdownNote", "Note", "Reroute", "PrimitiveNode"}


def unknown_types(info, saved):
    """Types de nodes du graphe enregistre absents du serveur (subgraphs compris)."""
    wf = json.loads(saved)
    subs = wf.get("definitions", {}).get("subgraphs", [])
    sub_ids = {s["id"] for s in subs}
    return sorted({n["type"] for g in [wf] + subs for n in g.get("nodes", [])
                   if n["type"] not in info and n["type"] not in sub_ids and n["type"] not in FRONT_ONLY})


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
    ap.add_argument("--dir", default=os.path.join(ROOT, "examples"), help="dossier des wf (defaut : examples/)")
    args = ap.parse_args()
    url = args.url.rstrip("/")
    info = json.loads(api(url, "/api/object_info"))
    wfs = sorted(glob.glob(os.path.join(args.dir, "*.json")))

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
                page.on("response", lambda r: errs.append(f"HTTP {r.status} {r.url}")
                        if r.status >= 400 and "/aioli/" in r.url else None)   # routes du pack
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
                    prompt = res["prompt"] or {}
                    unknown = unknown_types(info, res["saved"])
                    p = res["panel"]
                    if unknown and p is not None:
                        problems.append(f"nodes inconnus : {unknown}")
                    req = missing_required(info, prompt)
                    if req:
                        problems.append(f"entrees requises absentes : {req[:5]}")
                    if p is not None:
                        got = sorted(p["pairs"])
                        bad = [(k, nm) for k, nm in got if k >= len(PANEL_TO) or nm not in PANEL_TO[k]]
                        if len(got) != 9 or len({k for k, _ in got}) != 9 or bad:
                            problems.append(f"panneau -> subgraph : {got}")
                        if p["src"] != [["LoadImage", 0], ["LoadImage", 1]]:
                            problems.append(f"entrees image / mask du panneau : {p['src']}")
                        te = type_errors(info, p["uses"])
                        if te:
                            problems.append(f"types : {te}")
                        if p["masked_branch"]:
                            off = [n["inputs"].get("value") for n in prompt.values()
                                   if n.get("_meta", {}).get("title") == "MASKED LATENT"]
                            if p["masked_widget"] is not False or off != [False]:
                                problems.append(f"MASKED LATENT pas off par defaut (widget "
                                                f"{p['masked_widget']}, prompt {off})")
                    if args.shots:
                        os.makedirs(args.shots, exist_ok=True)
                        page.screenshot(path=os.path.join(args.shots, f"{mode}_{name[:-5]}.png"))
                    ok &= not problems
                    extra = (f" panneau: {len(p['pairs'])} liens, {len(p['uses'])} entrees alimentees" if p else
                             (f" (nodes d'autres packs absents ici : {len(unknown)})" if unknown else ""))
                    print(f"[{mode}] {name}: {'OK' if not problems else 'ECHEC ' + '; '.join(problems)}{extra}")
                page.context.close()
            browser.close()
    finally:
        if vue_before is not None:
            set_vue(vue_before)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
