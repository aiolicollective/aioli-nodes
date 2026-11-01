// 🎞️ Aioli Video Compare — lecteur A / B synchronise a l'image pres.
//
// Legacy (canvas LiteGraph) et Nodes 2.0 (Vue) : tout est dans un DOM widget
// (<canvas> + barre d'outils HTML) ; rien n'est dessine dans onDrawForeground.
// Lecture : proxys JPEG (temp de ComfyUI, servis par /view). Pause, pas a pas,
// zoom 1:1 et difference : images pleine resolution sans perte (PNG) via la
// route /aioli/video_compare/frame. Une horloge commune : si l'image A ou B de
// l'instant n'est pas encore decodee, l'horloge attend (jamais de decalage).

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "AioliVideoCompare";
const CACHE_BYTES = 768 * 1024 * 1024;   // budget des proxys decodes
const AHEAD = 36;                         // images prechargees devant
const PARALLEL = 6;                       // requetes simultanees
const MODES = [["wipe", "wipe"], ["side", "side by side"], ["diff", "difference"], ["toggle", "toggle"]];
const ZOOMS = [["fit", "fit"], ["1", "1:1"], ["2", "2:1"]];

function pad(i) { return String(i).padStart(6, "0"); }
function esc(t) {
  return String(t).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

// --- cache LRU d'ImageBitmap, budgete en octets --------------------------------
class FrameCache {
  constructor(budget) { this.budget = budget; this.map = new Map(); this.total = 0; }
  get(key) {
    const e = this.map.get(key);
    if (!e) return null;
    this.map.delete(key); this.map.set(key, e);            // plus recent
    return e.bmp;
  }
  put(key, bmp) {
    const bytes = bmp.width * bmp.height * 4;
    if (this.map.has(key)) return;
    this.map.set(key, { bmp, bytes });
    this.total += bytes;
    for (const [k, e] of this.map) {
      if (this.total <= this.budget) break;
      if (k === key) continue;
      e.bmp.close?.(); this.total -= e.bytes; this.map.delete(k);
    }
  }
  clear() { for (const e of this.map.values()) e.bmp.close?.(); this.map.clear(); this.total = 0; }
}

function installStyle() {
  if (document.getElementById("aioli-vc-style")) return;
  const st = document.createElement("style");
  st.id = "aioli-vc-style";
  st.textContent = `
    .aioli-vc { display:flex; flex-direction:column; gap:4px; width:100%; outline:none;
                font: 11px/1.3 sans-serif; color: var(--input-text, #ddd); }
    .aioli-vc canvas { width:100%; display:block; border-radius:4px; background:#111; touch-action:none; cursor:crosshair; }
    .aioli-vc-bar { display:flex; flex-wrap:wrap; align-items:center; gap:4px; }
    .aioli-vc-bar button, .aioli-vc-bar select { font:11px sans-serif; padding:1px 6px; min-height:20px;
                background: var(--comfy-input-bg, #222); color: var(--input-text, #ddd);
                border:1px solid var(--border-color, #444); border-radius:4px; cursor:pointer; }
    .aioli-vc-bar input[type=range] { flex:1 1 120px; min-width:80px; }
    .aioli-vc-info { opacity:.8; white-space:nowrap; }
    .aioli-vc-warn { color:#ffb347; }`;
  document.head.appendChild(st);
}

function setup(node) {
  installStyle();
  const root = document.createElement("div");
  root.className = "aioli-vc";
  root.tabIndex = 0;
  const canvas = document.createElement("canvas");
  const bar = document.createElement("div");
  bar.className = "aioli-vc-bar";
  const info = document.createElement("div");
  info.className = "aioli-vc-info";
  const warn = document.createElement("div");
  warn.className = "aioli-vc-warn";
  root.append(canvas, bar, info, warn);

  const S = {
    meta: null, playing: false, t: 0, speed: 1, loop: true, mode: "wipe", swap: false,
    showB: false, wipe: 0.5, zoom: "fit", pan: { x: 0, y: 0 }, gain: 1,
    cache: new FrameCache(CACHE_BYTES), pending: new Set(), inflight: 0, queue: [],
    full: { ia: -1, ib: -1, a: null, b: null, sid: null }, fullLoading: false,
    shown: { a: null, b: null, ia: -1, ib: -1, full: false }, dirty: true, alive: true, last: 0,
  };
  node._aioliVC = S;

  // --- barre d'outils ---------------------------------------------------------
  const btn = (label, title, fn) => {
    const b = document.createElement("button"); b.textContent = label; b.title = title;
    b.addEventListener("click", (e) => { e.stopPropagation(); fn(); root.focus(); });
    bar.appendChild(b); return b;
  };
  const sel = (opts, value, title, fn) => {
    const s = document.createElement("select"); s.title = title;
    for (const [v, l] of opts) { const o = document.createElement("option"); o.value = v; o.textContent = l; s.appendChild(o); }
    s.value = value;
    s.addEventListener("change", () => { fn(s.value); S.dirty = true; root.focus(); });
    s.addEventListener("pointerdown", (e) => e.stopPropagation());
    bar.appendChild(s); return s;
  };
  const bPlay = btn("▶", "play / pause (space)", () => togglePlay());
  btn("◀", "previous frame (←)", () => step(-1));
  btn("▶|", "next frame (→)", () => step(1));
  const scrub = document.createElement("input");
  scrub.type = "range"; scrub.min = "0"; scrub.max = "1000"; scrub.value = "0"; scrub.title = "scrub";
  scrub.addEventListener("input", () => {
    if (!S.meta) return;
    S.t = (Number(scrub.value) / 1000) * duration(); S.dirty = true;
  });
  scrub.addEventListener("pointerdown", (e) => e.stopPropagation());
  bar.appendChild(scrub);
  sel(MODES, S.mode, "comparison mode", (v) => { S.mode = v; });
  btn("A⇄B", "swap A and B", () => { S.swap = !S.swap; S.dirty = true; });
  sel(ZOOMS, S.zoom, "zoom (mouse wheel) — drag to pan", (v) => { S.zoom = v; S.pan = { x: 0, y: 0 }; });
  const gainSel = sel([["1", "diff ×1"], ["4", "diff ×4"], ["16", "diff ×16"]], "1", "difference gain", (v) => { S.gain = Number(v); });
  sel([["0.25", "×0.25"], ["0.5", "×0.5"], ["1", "×1"], ["2", "×2"]], "1", "speed", (v) => { S.speed = Number(v); });
  const loop = btn("loop", "loop on / off", () => { S.loop = !S.loop; loop.style.opacity = S.loop ? 1 : 0.5; });

  // --- temps et indices ---------------------------------------------------------
  const side = (k) => (S.swap ? (k === "a" ? "b" : "a") : k);      // A / B affiches -> clips
  const duration = () => {
    const { a, b } = S.meta; return Math.max(a.n / a.fps, b.n / b.fps);
  };
  const index = (clip, t) => {
    const m = S.meta[clip];
    return Math.max(0, Math.min(m.n - 1, Math.floor(t * m.fps + 1e-6)));
  };
  const masterFps = () => Math.max(S.meta.a.fps, S.meta.b.fps);

  function step(d) {
    if (!S.meta) return;
    S.playing = false; bPlay.textContent = "▶";
    const f = masterFps();
    const k = Math.floor(S.t * f + 1e-6) + d;
    const maxK = Math.ceil(duration() * f) - 1;
    S.t = (Math.max(0, Math.min(maxK, k)) + 0.5) / f;
    S.dirty = true;
  }
  function togglePlay() {
    if (!S.meta) return;
    S.playing = !S.playing; bPlay.textContent = S.playing ? "❚❚" : "▶";
    if (S.playing && S.t >= duration() - 1e-6) S.t = 0;
    S.dirty = true;
  }

  // --- chargement des proxys -------------------------------------------------------
  const key = (clip, i) => `${S.meta.sid}:${clip}:${i}`;
  function want(clip, i) {
    const k = key(clip, i);
    if (S.cache.get(k) || S.pending.has(k)) return;
    S.pending.add(k); S.queue.push([clip, i, k]);
  }
  function pump() {
    while (S.inflight < PARALLEL && S.queue.length) {
      const [clip, i, k] = S.queue.shift();
      const sid = S.meta.sid;
      S.inflight++;
      const url = `/view?filename=${clip}_${pad(i)}.jpg&type=temp&subfolder=${encodeURIComponent(S.meta.subfolder)}`;
      api.fetchApi(url).then((r) => (r.ok ? r.blob() : null))
        .then((blob) => (blob ? createImageBitmap(blob) : null))
        .then((bmp) => { if (bmp && S.meta && S.meta.sid === sid) { S.cache.put(k, bmp); S.dirty = true; } else bmp?.close?.(); })
        .catch(() => {})
        .finally(() => { S.inflight--; S.pending.delete(k); });
    }
  }
  function prefetch(t) {
    // ordre : l'image courante d'abord, puis devant (boucle comprise)
    for (const [, , k] of S.queue) S.pending.delete(k);    // file recalculee a chaque image
    S.queue = [];
    const f = masterFps(), D = duration();
    for (let s = 0; s < AHEAD; s++) {
      let tt = t + s / f;
      if (tt >= D) { if (!S.loop) break; tt -= D; }
      want("a", index("a", tt)); want("b", index("b", tt));
    }
  }

  // --- pleine resolution (pause, 1:1, difference) ---------------------------------
  function wantFull(ia, ib) {
    if (S.fullLoading || (S.full.sid === S.meta.sid && S.full.ia === ia && S.full.ib === ib)) return;
    const sid = S.meta.sid;
    S.fullLoading = true;
    const get = (clip, i) => api.fetchApi(`/aioli/video_compare/frame?sid=${sid}&side=${clip}&i=${i}`)
      .then((r) => (r.ok ? r.blob() : Promise.reject(new Error(r.status)))).then((b) => createImageBitmap(b));
    Promise.all([get("a", ia), get("b", ib)]).then(([a, b]) => {
      S.full.a?.close?.(); S.full.b?.close?.();
      S.full = { sid, ia, ib, a, b }; S.dirty = true;
    }).catch(() => {}).finally(() => { S.fullLoading = false; });
  }

  // --- dessin ------------------------------------------------------------------------
  function canvasSize() {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(50, canvas.clientWidth);
    const [W, H] = S.meta ? S.meta.display : [16, 9];
    const h = Math.round(Math.min(720, Math.max(180, w * H / W)));
    canvas.style.height = `${h}px`;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    }
    return { cw: canvas.width, ch: canvas.height, dpr };
  }
  // rectangle de destination de l'image (geometrie de A) dans une zone du canvas
  function placement(zx, zw, ch) {
    const [W, H] = S.meta.display;
    const s = S.zoom === "fit" ? Math.min(zw / W, ch / H) : Number(S.zoom);
    const x = zx + (zw - W * s) / 2 + S.pan.x;
    const y = (ch - H * s) / 2 + S.pan.y;
    return { x, y, w: W * s, h: H * s, s };
  }
  let diffCanvas = null;
  function draw(a, b, ia, ib, full) {
    const { cw, ch, dpr } = canvasSize();
    const ctx = canvas.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#111"; ctx.fillRect(0, 0, cw, ch);
    ctx.imageSmoothingEnabled = S.zoom === "fit";
    const A = S.swap ? b : a, B = S.swap ? a : b;          // ce qui est montre a gauche / a droite
    const la = S.meta[side("a")], lb = S.meta[side("b")];
    if (S.mode === "side") {
      const pa = placement(0, cw / 2, ch), pb = placement(cw / 2, cw / 2, ch);
      ctx.save(); ctx.beginPath(); ctx.rect(0, 0, cw / 2, ch); ctx.clip(); ctx.drawImage(A, pa.x, pa.y, pa.w, pa.h); ctx.restore();
      ctx.save(); ctx.beginPath(); ctx.rect(cw / 2, 0, cw / 2, ch); ctx.clip(); ctx.drawImage(B, pb.x, pb.y, pb.w, pb.h); ctx.restore();
      ctx.fillStyle = "#000"; ctx.fillRect(cw / 2 - dpr / 2, 0, dpr, ch);
    } else {
      const p = placement(0, cw, ch);
      if (S.mode === "diff") {
        if (!diffCanvas) diffCanvas = document.createElement("canvas");
        diffCanvas.width = cw; diffCanvas.height = ch;
        const d = diffCanvas.getContext("2d");
        d.imageSmoothingEnabled = ctx.imageSmoothingEnabled;
        d.clearRect(0, 0, cw, ch);
        d.drawImage(A, p.x, p.y, p.w, p.h);
        d.globalCompositeOperation = "difference";
        d.drawImage(B, p.x, p.y, p.w, p.h);
        d.globalCompositeOperation = "source-over";
        ctx.filter = S.gain !== 1 ? `brightness(${S.gain})` : "none";
        ctx.drawImage(diffCanvas, 0, 0);
        ctx.filter = "none";
      } else if (S.mode === "toggle") {
        ctx.drawImage(S.showB ? B : A, p.x, p.y, p.w, p.h);
      } else {                                              // wipe : A a gauche du trait
        const xw = Math.round(S.wipe * cw);
        ctx.drawImage(B, p.x, p.y, p.w, p.h);
        ctx.save(); ctx.beginPath(); ctx.rect(0, 0, xw, ch); ctx.clip(); ctx.drawImage(A, p.x, p.y, p.w, p.h); ctx.restore();
        ctx.fillStyle = "#fff"; ctx.fillRect(xw - dpr / 2, 0, dpr, ch);
      }
    }
    // etiquettes
    ctx.font = `${11 * dpr}px sans-serif`;
    const tag = (txt, x, y, align) => {
      ctx.textAlign = align;
      const w = ctx.measureText(txt).width + 8 * dpr;
      ctx.fillStyle = "rgba(0,0,0,.55)";
      ctx.fillRect(align === "left" ? x : x - w, y, w, 16 * dpr);
      ctx.fillStyle = "#fff";
      ctx.fillText(txt, align === "left" ? x + 4 * dpr : x - 4 * dpr, y + 12 * dpr);
    };
    const name = (letter, m) => `${letter}${m.label && m.label !== letter ? " · " + m.label : ""} · ${m.w}×${m.h}`;
    const nameA = name(S.swap ? "B" : "A", la), nameB = name(S.swap ? "A" : "B", lb);
    if (S.mode === "toggle") tag(S.showB ? nameB : nameA, 4 * dpr, 4 * dpr, "left");
    else if (S.mode === "diff") tag(`|${nameA}  −  ${nameB}|${S.gain !== 1 ? `  ×${S.gain}` : ""}`, 4 * dpr, 4 * dpr, "left");
    else { tag(nameA, 4 * dpr, 4 * dpr, "left"); tag(nameB, cw - 4 * dpr, 4 * dpr, "right"); }
    const fa = S.swap ? ib : ia, fb = S.swap ? ia : ib;
    info.textContent = `frame A ${fa + 1}/${la.n} · B ${fb + 1}/${lb.n} · ${S.t.toFixed(3)} s · ` +
      `${full ? "full resolution (lossless)" : "proxy " + S.meta.proxy.join("×")} · zoom ${S.zoom === "fit" ? "fit" : S.zoom + ":1"}`;
    scrub.value = String(Math.round((S.t / duration()) * 1000));
  }

  // --- boucle -----------------------------------------------------------------------------
  function frame(now) {
    if (!S.alive) return;
    requestAnimationFrame(frame);
    const dt = S.last ? Math.min(0.25, (now - S.last) / 1000) : 0;
    S.last = now;
    if (!S.meta || !root.isConnected) return;
    const D = duration();
    let t = S.t;
    if (S.playing) {
      t = S.t + dt * S.speed;
      if (t >= D) { if (S.loop) t -= D; else { t = D - 1e-6; S.playing = false; bPlay.textContent = "▶"; } }
    }
    const ia = index("a", t), ib = index("b", t);
    const a = S.cache.get(key("a", ia)), b = S.cache.get(key("b", ib));
    prefetch(t);
    pump();
    if (a && b) {
      if (t !== S.t || S.dirty || S.shown.ia !== ia || S.shown.ib !== ib) {
        S.t = t;                                          // l'horloge n'avance que si la paire est prete
        const needFull = !S.playing && (S.zoom !== "fit" || S.mode === "diff");
        if (needFull) wantFull(ia, ib);
        const useFull = !S.playing && S.full.sid === S.meta.sid && S.full.ia === ia && S.full.ib === ib;
        if (S.dirty || S.shown.ia !== ia || S.shown.ib !== ib || S.shown.full !== useFull) {
          draw(useFull ? S.full.a : a, useFull ? S.full.b : b, ia, ib, useFull);
          S.shown = { ia, ib, full: useFull };
          S.dirty = false;
        }
      }
    }
  }
  requestAnimationFrame(frame);

  // --- souris / clavier -------------------------------------------------------------------
  let drag = null;
  canvas.addEventListener("pointerdown", (e) => {
    e.stopPropagation(); e.preventDefault(); root.focus();
    canvas.setPointerCapture(e.pointerId);
    const panMode = e.shiftKey || (S.mode !== "wipe" && S.zoom !== "fit");
    drag = { x: e.clientX, y: e.clientY, pan: { ...S.pan }, panMode };
    if (S.mode === "toggle" && !panMode) { S.showB = !S.showB; S.dirty = true; }
    if (S.mode === "wipe" && !panMode) moveWipe(e);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    e.stopPropagation();
    if (drag.panMode) {
      const dpr = window.devicePixelRatio || 1;
      S.pan = { x: drag.pan.x + (e.clientX - drag.x) * dpr, y: drag.pan.y + (e.clientY - drag.y) * dpr };
      S.dirty = true;
    } else if (S.mode === "wipe") moveWipe(e);
  });
  const end = (e) => { drag = null; try { canvas.releasePointerCapture(e.pointerId); } catch (_) {} };
  canvas.addEventListener("pointerup", end);
  canvas.addEventListener("pointercancel", end);
  function moveWipe(e) {
    const r = canvas.getBoundingClientRect();
    S.wipe = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)); S.dirty = true;
  }
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault(); e.stopPropagation();
    const order = ZOOMS.map((z) => z[0]);
    const k = Math.max(0, Math.min(order.length - 1, order.indexOf(S.zoom) + (e.deltaY < 0 ? 1 : -1)));
    S.zoom = order[k]; if (S.zoom === "fit") S.pan = { x: 0, y: 0 };
    bar.querySelectorAll("select")[1].value = S.zoom; S.dirty = true;
  }, { passive: false });
  root.addEventListener("keydown", (e) => {
    const k = e.key;
    if (k === " ") togglePlay();
    else if (k === "ArrowLeft") step(-1);
    else if (k === "ArrowRight") step(1);
    else if (k === "t" || k === "T") { S.showB = !S.showB; S.dirty = true; }
    else return;
    e.preventDefault(); e.stopPropagation();
  });

  // --- widget --------------------------------------------------------------------------------
  const widget = node.addDOMWidget("video_compare", "AIOLI_VIDEO_COMPARE", root, {
    serialize: false,
    hideOnZoom: false,
    getMinHeight: () => (canvas.clientHeight || 220) + bar.offsetHeight + info.offsetHeight + warn.offsetHeight + 16,
  });
  widget.serialize = false;
  widget.serializeValue = () => undefined;
  const ro = new ResizeObserver(() => { S.dirty = true; });
  ro.observe(root);

  S.load = (meta) => {
    S.cache.clear(); S.queue = []; S.pending.clear();
    S.full.a?.close?.(); S.full.b?.close?.();
    S.full = { ia: -1, ib: -1, a: null, b: null, sid: null };
    S.meta = meta; S.t = 0; S.playing = false; bPlay.textContent = "▶";
    S.shown = { ia: -1, ib: -1, full: false }; S.dirty = true;
    warn.innerHTML = (meta.warnings || []).map((w) => `⚠ ${esc(w)}`).join("<br>");
    gainSel.disabled = false;
    if (node.size[0] < 520) node.setSize?.([640, node.size[1]]);
    requestAnimationFrame(() => {
      const h = node.computeSize?.()[1];
      if (h && node.size[1] < h) node.setSize?.([node.size[0], h]);
      node.setDirtyCanvas?.(true, true);
    });
  };

  const onRemoved = node.onRemoved;
  node.onRemoved = function () {
    S.alive = false; ro.disconnect(); S.cache.clear();
    S.full.a?.close?.(); S.full.b?.close?.();
    return onRemoved?.apply(this, arguments);
  };
  if (!node.size || node.size[0] < 520) node.setSize?.([640, node.size?.[1] ?? 480]);
}

app.registerExtension({
  name: "aioli.VideoCompare",
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
      const meta = message?.aioli_vcompare?.[0];
      if (meta && this._aioliVC) this._aioliVC.load(meta);
    };
  },
});
