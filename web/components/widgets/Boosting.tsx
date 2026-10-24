"use client";
import { useMemo, useState } from "react";
import { Plane, dot, drawGrid, fmt, rng, textColor, useCanvas } from "./lib";

type Stump = { t: number; left: number; right: number };

/** Best single split (regression stump) on 1-D data minimising squared error. */
function fitStump(xs: number[], r: number[]): Stump {
  const order = xs.map((_, i) => i).sort((a, b) => xs[a] - xs[b]);
  const total = r.reduce((a, b) => a + b, 0);
  let best = { sse: Infinity, t: 0, left: 0, right: 0 }, sl = 0;
  for (let k = 0; k < order.length - 1; k++) {
    sl += r[order[k]];
    const nl = k + 1, nr = order.length - nl;
    const ml = sl / nl, mr = (total - sl) / nr;
    const sse = -(nl * ml * ml + nr * mr * mr); // equivalent objective (constant term dropped)
    if (sse < best.sse) best = { sse, t: (xs[order[k]] + xs[order[k + 1]]) / 2, left: ml, right: mr };
  }
  return best;
}

export default function Boosting() {
  const [lr, setLr] = useState(0.3);
  const [stage, setStage] = useState(1);
  const [seed, setSeed] = useState(2);
  const data = useMemo(() => {
    const r = rng(seed);
    const xs = Array.from({ length: 80 }, () => r.next() * 6);
    return { xs, ys: xs.map((x) => Math.sin(x) + 0.3 * x + 0.25 * r.normal()) };
  }, [seed]);
  const model = useMemo(() => {
    const f0 = data.ys.reduce((a, b) => a + b, 0) / data.ys.length;
    const stumps: Stump[] = [];
    let F = data.xs.map(() => f0);
    const residualsAt: number[][] = [];
    for (let m = 0; m < 100; m++) {
      const r = data.ys.map((y, i) => y - F[i]); // negative gradient of ½(y − F)²
      residualsAt.push(r);
      const s = fitStump(data.xs, r);
      stumps.push(s);
      F = F.map((v, i) => v + lr * (data.xs[i] <= s.t ? s.left : s.right));
    }
    const predict = (x: number, upto: number) => stumps.slice(0, upto).reduce((v, s) => v + lr * (x <= s.t ? s.left : s.right), f0);
    return { f0, stumps, residualsAt, predict };
  }, [data, lr]);
  const mse = data.xs.reduce((s, x, i) => s + (data.ys[i] - model.predict(x, stage)) ** 2, 0) / data.xs.length;

  const P1 = new Plane(500, 260, 0, 6, -1.5, 3.2);
  const fitRef = useCanvas((ctx) => {
    drawGrid(ctx, P1, 1);
    data.xs.forEach((x, i) => dot(ctx, P1.px(x), P1.py(data.ys[i]), "#4263eb", 3.5));
    ctx.strokeStyle = "#e8590c"; ctx.lineWidth = 2.5; ctx.beginPath();
    for (let i = 0; i <= 400; i++) { const x = (6 * i) / 400; const y = model.predict(x, stage); i ? ctx.lineTo(P1.px(x), P1.py(y)) : ctx.moveTo(P1.px(x), P1.py(y)); }
    ctx.stroke();
    ctx.fillStyle = textColor(); ctx.font = "12px ui-sans-serif"; ctx.fillText(`F_${stage}(x) = F₀ + η·Σ h_m(x)   (${stage} stumps)`, 8, 16);
  }, [data, model, stage], 500, 260);

  const P2 = new Plane(500, 180, 0, 6, -1.6, 1.6);
  const resRef = useCanvas((ctx) => {
    drawGrid(ctx, P2, 1);
    const r = model.residualsAt[stage - 1];
    data.xs.forEach((x, i) => dot(ctx, P2.px(x), P2.py(r[i]), "#868e96", 3));
    const s = model.stumps[stage - 1];
    ctx.strokeStyle = "#2b8a3e"; ctx.lineWidth = 3; ctx.beginPath();
    ctx.moveTo(P2.px(0), P2.py(s.left)); ctx.lineTo(P2.px(s.t), P2.py(s.left)); ctx.lineTo(P2.px(s.t), P2.py(s.right)); ctx.lineTo(P2.px(6), P2.py(s.right)); ctx.stroke();
    ctx.fillStyle = textColor(); ctx.font = "12px ui-sans-serif"; ctx.fillText(`residuals before stage ${stage} (grey) and the stump h_${stage} fit to them (green)`, 8, 16);
  }, [data, model, stage], 500, 180);

  return (
    <div className="widget">
      <h3>Gradient boosting, stage by stage</h3>
      <p className="muted" style={{ marginTop: 0 }}>Each stage fits a tiny tree (a stump) to the <b>residuals</b> of the model so far, then adds a shrunken copy of it. Step through the stages and watch the residuals shrink.</p>
      <div className="controls">
        <label>stage m = {stage} <input type="range" min={1} max={100} value={stage} onChange={(e) => setStage(Number(e.target.value))} /></label>
        <label>learning rate η = {lr.toFixed(2)} <input type="range" min={0.05} max={1} step={0.05} value={lr} onChange={(e) => setLr(Number(e.target.value))} /></label>
        <button className="btn" onClick={() => setSeed(seed + 1)}>New data</button>
        <span className="stat">training MSE = {fmt(mse)}</span>
      </div>
      <canvas ref={fitRef} />
      <canvas ref={resRef} style={{ marginTop: 8 }} />
      <p className="muted" style={{ fontSize: "0.85rem" }}>With η = 1 the model fits fast but jumpy; small η needs more stages but generalises better (shrinkage). Go to stage 100: with enough stages, boosting starts fitting the noise.</p>
    </div>
  );
}
