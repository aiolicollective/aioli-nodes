// 📏 Aioli Resolution Panel — schema de la chaine de resolution, en direct.
//
// Legacy (canvas LiteGraph) et Nodes 2.0 (Vue) : tout passe par un DOM widget
// (un <div> avec un <canvas> et du texte HTML). Rien n'est dessine dans
// onDrawForeground, qui n'est pas appele en Nodes 2.0.
//
// Les calculs ne sont PAS faits ici : la route /aioli/resolution_plan appelle
// les memes fonctions Python que les nodes (nodes/resolution_plan.py). Ce
// fichier ne fait que lire les reglages, appeler la route et dessiner.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "AioliResolutionPanel";
const PARAMS = ["scale", "ratio", "regenerate_all", "resolution", "multiple",
                "force_square", "force_target_downscale", "padding", "context"];
const PALETTE = ["#ff3333", "#33ccff", "#66ff4d", "#ffd91a", "#e64dff", "#ff8c00", "#33ffcc", "#ff66b3"];
const POLL_MS = 400;
const DIAGRAM_H = 210;

function widgetValue(node, name) {
  const w = node.widgets?.find((w) => w.name === name);
  return w ? w.value : undefined;
}

function inputSlot(node, name) {
  return node.inputs ? node.inputs.findIndex((i) => i.name === name) : -1;
}

function originNode(node, name) {
  const slot = inputSlot(node, name);
  if (slot < 0 || !node.isInputConnected?.(slot)) return null;
  return node.getInputNode?.(slot) ?? null;
}

// Image source : un LoadImage (ou tout node avec un widget "image" = fichier).
function sourceImage(node) {
  const src = originNode(node, "image");
  if (!src) return { status: "none" };
  const w = src.widgets?.find((w) => w.name === "image");
  if (!w || !w.value) return { status: "unknown" };
  const maskSrc = originNode(node, "mask");
  return { status: "file", name: String(w.value), useMask: !!maskSrc && maskSrc.id === src.id };
}

function fmt(n) { return Math.round(n).toString(); }
function size(a) { return `${fmt(a[0])}×${fmt(a[1])}`; }

function hatch(ctx, x, y, w, h, color) {
  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, w, h);
  ctx.clip();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  for (let k = -h; k < w; k += 7) {
    ctx.beginPath();
    ctx.moveTo(x + k, y + h);
    ctx.lineTo(x + k + h, y);
    ctx.stroke();
  }
  ctx.restore();
}

function draw(state) {
  const { canvas, text, plan, message } = state;
  const dpr = window.devicePixelRatio || 1;
  const cw = Math.max(50, canvas.clientWidth);
  const ch = DIAGRAM_H;
  if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
    canvas.width = Math.round(cw * dpr);
    canvas.height = Math.round(ch * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);
  ctx.fillStyle = "#1b1b1f";
  ctx.fillRect(0, 0, cw, ch);

  if (!plan) {
    ctx.fillStyle = "#aaa";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(message || "…", cw / 2, ch / 2);
    text.innerHTML = "";
    return;
  }

  // Monde : le canevas apres outpaint + tous les crops (bandes hors image comprises).
  const [PW, PH] = plan.padded;
  let x0 = 0, y0 = 0, x1 = PW, y1 = PH;
  for (const r of plan.regions) {
    const [cx, cy, cwr, chr] = r.crop;
    x0 = Math.min(x0, cx); y0 = Math.min(y0, cy);
    x1 = Math.max(x1, cx + cwr); y1 = Math.max(y1, cy + chr);
  }
  const m = 10;
  const s = Math.min((cw - 2 * m) / (x1 - x0), (ch - 2 * m) / (y1 - y0));
  const ox = m + ((cw - 2 * m) - (x1 - x0) * s) / 2 - x0 * s;
  const oy = m + ((ch - 2 * m) - (y1 - y0) * s) / 2 - y0 * s;
  const X = (v) => ox + v * s, Y = (v) => oy + v * s;

  // Canevas d'outpaint (bandes a generer, hachures grises) puis l'image source.
  const [pl, , pt] = plan.pads;
  const [sw, sh] = plan.scaled;
  ctx.fillStyle = "#3a3a40";
  ctx.fillRect(X(0), Y(0), PW * s, PH * s);
  hatch(ctx, X(0), Y(0), PW * s, PH * s, "#5a5a62");
  ctx.fillStyle = "#33465a";
  ctx.fillRect(X(pl), Y(pt), sw * s, sh * s);
  ctx.strokeStyle = "#8fb3d9";
  ctx.lineWidth = 1;
  ctx.strokeRect(X(pl) + 0.5, Y(pt) + 0.5, sw * s - 1, sh * s - 1);
  ctx.fillStyle = "#8fb3d9";
  ctx.font = "11px sans-serif";
  ctx.textAlign = "left";
  ctx.fillText(`image ${size(plan.scaled)}`, X(pl) + 4, Y(pt) + 13);

  // Regions : crop (pointilles), bandes du crop hors canevas (hachures), bbox (trait plein).
  plan.regions.forEach((r, k) => {
    const col = PALETTE[k % PALETTE.length];
    const [cx, cy, cwr, chr] = r.crop;
    const [bx, by, bw, bh] = r.bbox;
    // bandes : partie du crop hors du canevas
    const ix0 = Math.max(cx, 0), iy0 = Math.max(cy, 0);
    const ix1 = Math.min(cx + cwr, PW), iy1 = Math.min(cy + chr, PH);
    const parts = [[cx, cy, cwr, iy0 - cy], [cx, iy1, cwr, cy + chr - iy1],
                   [cx, iy0, ix0 - cx, iy1 - iy0], [ix1, iy0, cx + cwr - ix1, iy1 - iy0]];
    for (const [px, py, pw, ph] of parts) {
      if (pw > 0 && ph > 0) hatch(ctx, X(px), Y(py), pw * s, ph * s, col + "99");
    }
    ctx.setLineDash([5, 3]);
    ctx.strokeStyle = col;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(X(cx), Y(cy), cwr * s, chr * s);
    ctx.setLineDash([]);
    ctx.lineWidth = 1;
    ctx.strokeRect(X(bx), Y(by), bw * s, bh * s);
    ctx.fillStyle = col;
    ctx.font = "bold 11px sans-serif";
    ctx.fillText(`R${r.index}`, X(cx) + 3, Y(cy) + 12);
  });

  // Texte : une ligne pour la chaine, une par region, puis les alertes.
  const st = plan.settings || {};
  let chain = `source ${size(plan.source)}`;
  if (sw !== plan.source[0] || sh !== plan.source[1]) chain += ` → ×${st.scale} = ${size(plan.scaled)}`;
  if (PW !== sw || PH !== sh) chain += ` → ${st.ratio} = ${size(plan.padded)} (original ${Math.round(plan.coverage)} %)`;
  const rows = [`<div class="aioli-rp-chain">${chain}</div>`];
  plan.regions.forEach((r, k) => {
    const col = PALETTE[k % PALETTE.length];
    const sc = r.model_scale;
    const rs = !r.resize ? "pas de resize"
      : sc >= 1 ? `↑ ×${sc.toFixed(2)}` : `↓ ×${sc.toFixed(2)} (1/${(1 / sc).toFixed(1)})`;
    const bands = r.band_pct ? ` · bandes ${Math.round(r.band_pct)} %` : "";
    rows.push(`<div><span class="aioli-rp-sw" style="background:${col}"></span>` +
              `<b>R${r.index}</b> ${r.kind} · crop ${size(r.crop.slice(2))} → <b>${size(r.model)}</b>` +
              ` · ${rs}${bands} <span class="aioli-rp-dim">(bbox ${size(r.bbox.slice(2))})</span></div>`);
  });
  for (const w of plan.warnings || []) rows.push(`<div class="aioli-rp-warn">⚠ ${w}</div>`);
  rows.push(`<div class="aioli-rp-origin">${plan.origin === "run" ? "valeurs du dernier run" : "aperçu avant run"}</div>`);
  text.innerHTML = rows.join("");
}

function installStyle() {
  if (document.getElementById("aioli-rp-style")) return;
  const st = document.createElement("style");
  st.id = "aioli-rp-style";
  st.textContent = `
    .aioli-rp { display:flex; flex-direction:column; gap:4px; width:100%; box-sizing:border-box;
                font: 11px/1.35 sans-serif; color: var(--input-text, #ddd); }
    .aioli-rp canvas { width:100%; height:${DIAGRAM_H}px; border-radius:4px; display:block; }
    .aioli-rp-text { display:flex; flex-direction:column; gap:1px; padding:0 2px; }
    .aioli-rp-chain { opacity:.9; }
    .aioli-rp-sw { display:inline-block; width:8px; height:8px; border-radius:2px; margin-right:4px; }
    .aioli-rp-warn { color:#ffb347; }
    .aioli-rp-origin { opacity:.5; font-size:10px; }
    .aioli-rp-dim { opacity:.55; }`;
  document.head.appendChild(st);
}

function setup(node) {
  installStyle();
  const root = document.createElement("div");
  root.className = "aioli-rp";
  const canvas = document.createElement("canvas");
  const text = document.createElement("div");
  text.className = "aioli-rp-text";
  root.append(canvas, text);

  const state = { canvas, text, plan: null, message: "branche l'image source (LoadImage)",
                  sig: null, timer: null, seq: 0 };
  node._aioliRP = state;

  const widget = node.addDOMWidget("resolution_view", "AIOLI_RESOLUTION_VIEW", root, {
    serialize: false,
    hideOnZoom: false,
    getMinHeight: () => DIAGRAM_H + 30 + root.querySelector(".aioli-rp-text").offsetHeight,
  });
  widget.serializeValue = () => undefined;

  const redraw = () => draw(state);
  new ResizeObserver(redraw).observe(root);

  async function refresh(src, settings) {
    const seq = ++state.seq;
    try {
      const resp = await api.fetchApi("/aioli/resolution_plan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: src.name, use_mask: src.useMask, settings }),
      });
      const data = await resp.json();
      if (seq !== state.seq) return;          // une reponse plus recente est partie
      if (data.error) { state.plan = null; state.message = data.error; }
      else { state.plan = data; }
    } catch (e) {
      if (seq !== state.seq) return;
      state.plan = null;
      state.message = `apercu indisponible (${e})`;
    }
    redraw();
    node.setDirtyCanvas?.(true, true);
  }

  // Sondage des reglages : marche quel que soit le chemin du changement
  // (widget legacy, composant Vue, lien), sans dependre d'un hook du frontend.
  function poll() {
    const settings = Object.fromEntries(PARAMS.map((p) => [p, widgetValue(node, p)]));
    const src = sourceImage(node);
    const sig = JSON.stringify([settings, src]);
    if (sig === state.sig) return;
    state.sig = sig;
    if (src.status === "none") { state.plan = null; state.message = "branche l'image source (LoadImage)"; redraw(); return; }
    if (src.status === "unknown") { state.plan = null; state.message = "taille connue après run (source sans fichier)"; redraw(); return; }
    clearTimeout(state.timer);
    state.timer = setTimeout(() => refresh(src, settings), 120);
  }
  state.interval = setInterval(poll, POLL_MS);
  poll();

  const onRemoved = node.onRemoved;
  node.onRemoved = function () {
    clearInterval(state.interval);
    clearTimeout(state.timer);
    return onRemoved?.apply(this, arguments);
  };
  if (!node.size || node.size[0] < 420) node.setSize?.([420, node.size?.[1] ?? 400]);
}

app.registerExtension({
  name: "aioli.ResolutionPanel",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      setup(this);
      return r;
    };
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const p = message?.aioli_plan?.[0];
      if (p && this._aioliRP) {
        this._aioliRP.plan = p;
        draw(this._aioliRP);
      }
    };
  },
});
