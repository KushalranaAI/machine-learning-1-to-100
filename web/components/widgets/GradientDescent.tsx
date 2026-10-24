"use client";
import { useMemo, useState } from "react";
import { Plane, canvasPos, fmt, lineChart, useCanvas, useLoop } from "./lib";

type Fn = { name: string; f: (x: number, y: number) => number; g: (x: number, y: number) => [number, number]; box: [number, number, number, number]; start: [number, number]; min: number };

const FUNCTIONS: Record<string, Fn> = {
  bowl: { name: "Round bowl  x² + y²", f: (x, y) => x * x + y * y, g: (x, y) => [2 * x, 2 * y], box: [-2, 2, -2, 2], start: [-1.7, 1.5], min: 0 },
  valley: { name: "Elongated valley  ½(x² + 25y²)", f: (x, y) => 0.5 * (x * x + 25 * y * y), g: (x, y) => [x, 25 * y], box: [-2, 2, -1.2, 1.2], start: [-1.8, 0.9], min: 0 },
  rosen: { name: "Rosenbrock banana", f: (x, y) => (1 - x) ** 2 + 10 * (y - x * x) ** 2, g: (x, y) => [-2 * (1 - x) - 40 * x * (y - x * x), 20 * (y - x * x)], box: [-2, 2, -1, 3], start: [-1.5, 2.5], min: 0 },
  wavy: { name: "Non-convex (local minima)", f: (x, y) => x * x + y * y + 1.5 * Math.sin(3 * x) * Math.sin(3 * y), g: (x, y) => [2 * x + 4.5 * Math.cos(3 * x) * Math.sin(3 * y), 2 * y + 4.5 * Math.sin(3 * x) * Math.cos(3 * y)], box: [-2, 2, -2, 2], start: [1.6, 1.2], min: -1.5 },
};

type Opt = "gd" | "momentum" | "rmsprop" | "adam";

function step(opt: Opt, s: State, fn: Fn, lr: number): State {
  const [x, y] = s.path[s.path.length - 1];
  const g = fn.g(x, y);
  const t = s.t + 1;
  let v = s.v, m = s.m, sq = s.sq;
  let d: [number, number];
  if (opt === "momentum") {
    v = [0.9 * v[0] + g[0], 0.9 * v[1] + g[1]];
    d = v;
  } else if (opt === "rmsprop") {
    sq = [0.9 * sq[0] + 0.1 * g[0] ** 2, 0.9 * sq[1] + 0.1 * g[1] ** 2];
    d = [g[0] / (Math.sqrt(sq[0]) + 1e-8), g[1] / (Math.sqrt(sq[1]) + 1e-8)];
  } else if (opt === "adam") {
    m = [0.9 * m[0] + 0.1 * g[0], 0.9 * m[1] + 0.1 * g[1]];
    sq = [0.999 * sq[0] + 0.001 * g[0] ** 2, 0.999 * sq[1] + 0.001 * g[1] ** 2];
    const mh = [m[0] / (1 - 0.9 ** t), m[1] / (1 - 0.9 ** t)];
    const vh = [sq[0] / (1 - 0.999 ** t), sq[1] / (1 - 0.999 ** t)];
    d = [mh[0] / (Math.sqrt(vh[0]) + 1e-8), mh[1] / (Math.sqrt(vh[1]) + 1e-8)];
  } else d = g;
  const nx: [number, number] = [x - lr * d[0], y - lr * d[1]];
  const diverged = !Number.isFinite(nx[0]) || Math.abs(nx[0]) > 1e6 || Math.abs(nx[1]) > 1e6;
  return { path: [...s.path, nx], t, v, m, sq, losses: [...s.losses, fn.f(nx[0], nx[1])], diverged };
}

type State = { path: [number, number][]; t: number; v: [number, number]; m: [number, number]; sq: [number, number]; losses: number[]; diverged: boolean };
const init = (fn: Fn, p = fn.start): State => ({ path: [p], t: 0, v: [0, 0], m: [0, 0], sq: [0, 0], losses: [fn.f(p[0], p[1])], diverged: false });

const W = 520, H = 380;

export default function GradientDescent() {
  const [fnKey, setFnKey] = useState("valley");
  const fn = FUNCTIONS[fnKey];
  const [opt, setOpt] = useState<Opt>("gd");
  const [lr, setLr] = useState(0.05);
  const [state, setState] = useState(() => init(fn));
  const [running, setRunning] = useState(false);
  const P = useMemo(() => new Plane(W, H, ...fn.box), [fn]);

  const heat = useMemo(() => {
    if (typeof document === "undefined") return null;
    const c = document.createElement("canvas");
    c.width = W; c.height = H;
    const ctx = c.getContext("2d")!;
    const img = ctx.createImageData(W, H);
    let hi = -Infinity;
    const vals = new Float32Array(W * H);
    for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
      const v = Math.log1p(fn.f(P.dx(i), P.dy(j)) - fn.min);
      vals[j * W + i] = v; hi = Math.max(hi, v);
    }
    for (let k = 0; k < W * H; k++) {
      const t = vals[k] / hi;
      const band = Math.abs(((vals[k] * 6) % 1) - 0.5) < 0.04 ? 0.82 : 1; // contour lines
      img.data[4 * k] = (40 + 200 * t) * band;
      img.data[4 * k + 1] = (90 + 120 * (1 - t)) * band;
      img.data[4 * k + 2] = (200 - 150 * t) * band;
      img.data[4 * k + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return c;
  }, [fn, P]);

  const ref = useCanvas((ctx) => {
    if (heat) ctx.drawImage(heat, 0, 0, W, H);
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
    ctx.beginPath();
    state.path.forEach(([x, y], i) => (i ? ctx.lineTo(P.px(x), P.py(y)) : ctx.moveTo(P.px(x), P.py(y))));
    ctx.stroke();
    state.path.forEach(([x, y], i) => {
      ctx.beginPath(); ctx.arc(P.px(x), P.py(y), i === state.path.length - 1 ? 5 : 2.2, 0, 7);
      ctx.fillStyle = i === state.path.length - 1 ? "#ffd43b" : "#fff"; ctx.fill();
    });
  }, [state, heat, P], W, H);

  const lossRef = useCanvas((ctx, w, h) => lineChart(ctx, [{ values: state.losses, color: "#e8590c" }], 30, 6, w - 36, h - 14, { log: true, title: "loss (log scale)" }), [state], 260, 160);

  useLoop(running, () => {
    setState((s) => {
      if (s.diverged || s.t >= 400) { setRunning(false); return s; }
      return step(opt, s, fn, lr);
    });
  }, 30);

  const reset = (p?: [number, number]) => { setRunning(false); setState(init(fn, p)); };
  const last = state.path[state.path.length - 1];

  return (
    <div className="widget">
      <h3>Gradient descent explorer</h3>
      <p className="muted" style={{ marginTop: 0 }}>Click the surface to choose a starting point, pick an optimiser and press play. Darker = lower loss.</p>
      <div className="controls">
        <label>Surface <select value={fnKey} onChange={(e) => { setFnKey(e.target.value); setRunning(false); setState(init(FUNCTIONS[e.target.value])); }}>
          {Object.entries(FUNCTIONS).map(([k, v]) => <option key={k} value={k}>{v.name}</option>)}
        </select></label>
        <label>Optimiser <select value={opt} onChange={(e) => { setOpt(e.target.value as Opt); reset(); }}>
          <option value="gd">Gradient descent</option><option value="momentum">Momentum (β=0.9)</option>
          <option value="rmsprop">RMSProp</option><option value="adam">Adam</option>
        </select></label>
        <label>learning rate η = {lr.toFixed(3)} <input type="range" min={-3} max={0} step={0.01} value={Math.log10(lr)} onChange={(e) => { setLr(10 ** Number(e.target.value)); reset(); }} /></label>
      </div>
      <div className="controls">
        <button className="btn primary" onClick={() => setRunning(!running)}>{running ? "⏸ Pause" : "▶ Play"}</button>
        <button className="btn" onClick={() => setState((s) => step(opt, s, fn, lr))}>Step</button>
        <button className="btn" onClick={() => reset()}>Reset</button>
        <span className="stat">step {state.t} · θ = ({fmt(last[0])}, {fmt(last[1])}) · loss = {fmt(state.losses[state.losses.length - 1])} {state.diverged && "· 💥 diverged!"}</span>
      </div>
      <div style={{ display: "flex", gap: 14, flexWrap: "wrap", alignItems: "flex-start" }}>
        <canvas ref={ref} onClick={(e) => { const [x, y] = canvasPos(e, W, H); reset([P.dx(x), P.dy(y)]); }} style={{ cursor: "crosshair" }} />
        <div>
          <canvas ref={lossRef} />
          <p className="muted" style={{ maxWidth: 260, fontSize: "0.85rem" }}>
            Try the <b>elongated valley</b>: plain GD zig-zags and diverges above η ≈ 0.08 (= 2/λ<sub>max</sub>). Momentum and Adam cope much better.
            On the <b>non-convex</b> surface, different starting points end in different local minima.
          </p>
        </div>
      </div>
    </div>
  );
}
