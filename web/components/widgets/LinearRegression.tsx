"use client";
import { useState } from "react";
import { Plane, canvasPos, dot, drawGrid, fmt, lineChart, rng, textColor, useCanvas, useLoop } from "./lib";

const W = 520, H = 380;
const P = new Plane(W, H, 0, 10, 0, 10);

function initialPoints(seed = 3): [number, number][] {
  const r = rng(seed);
  return Array.from({ length: 18 }, () => {
    const x = 0.5 + 9 * r.next();
    return [x, 1.2 + 0.7 * x + 1.1 * r.normal()] as [number, number];
  });
}

/** Closed form: w = cov(x,y)/var(x), b = ȳ - w x̄ (the normal equation for one feature). */
function ols(pts: [number, number][]) {
  const n = pts.length;
  const mx = pts.reduce((s, p) => s + p[0], 0) / n;
  const my = pts.reduce((s, p) => s + p[1], 0) / n;
  let sxy = 0, sxx = 0;
  for (const [x, y] of pts) { sxy += (x - mx) * (y - my); sxx += (x - mx) ** 2; }
  const w = sxx > 0 ? sxy / sxx : 0;
  return { w, b: my - w * mx };
}

const mse = (pts: [number, number][], w: number, b: number) => pts.reduce((s, [x, y]) => s + (y - (w * x + b)) ** 2, 0) / Math.max(pts.length, 1);

export default function LinearRegression() {
  const [pts, setPts] = useState(initialPoints);
  const [model, setModel] = useState({ w: 0, b: 5 });
  const [losses, setLosses] = useState<number[]>([]);
  const [lr, setLr] = useState(0.01);
  const [running, setRunning] = useState(false);
  const [showSquares, setShowSquares] = useState(true);
  const best = pts.length >= 2 ? ols(pts) : { w: 0, b: 0 };

  useLoop(running, () => {
    setModel(({ w, b }) => {
      // gradients of MSE: ∂/∂w = -(2/n) Σ x(y - ŷ),  ∂/∂b = -(2/n) Σ (y - ŷ)
      let gw = 0, gb = 0;
      for (const [x, y] of pts) { const r = y - (w * x + b); gw += -2 * x * r; gb += -2 * r; }
      gw /= pts.length; gb /= pts.length;
      const next = { w: w - lr * gw, b: b - lr * gb };
      if (!Number.isFinite(next.w) || Math.abs(next.w) > 1e4) { setRunning(false); return { w, b }; }
      setLosses((l) => [...l.slice(-299), mse(pts, next.w, next.b)]);
      return next;
    });
  }, 40);

  const ref = useCanvas((ctx) => {
    drawGrid(ctx, P, 1);
    const line = (w: number, b: number, color: string, dash: number[] = []) => {
      ctx.setLineDash(dash); ctx.strokeStyle = color; ctx.lineWidth = 2.2;
      ctx.beginPath(); ctx.moveTo(P.px(0), P.py(b)); ctx.lineTo(P.px(10), P.py(10 * w + b)); ctx.stroke(); ctx.setLineDash([]);
    };
    for (const [x, y] of pts) {
      const yh = model.w * x + model.b;
      if (showSquares) {
        const side = Math.abs(P.py(y) - P.py(yh));
        ctx.fillStyle = "rgba(232,89,12,0.12)";
        ctx.fillRect(P.px(x), Math.min(P.py(y), P.py(yh)), side, side);
      }
      ctx.strokeStyle = "rgba(232,89,12,0.7)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(P.px(x), P.py(y)); ctx.lineTo(P.px(x), P.py(yh)); ctx.stroke();
    }
    if (pts.length >= 2) line(best.w, best.b, "rgba(43,138,62,0.9)", [6, 5]);
    line(model.w, model.b, "#4263eb");
    for (const [x, y] of pts) dot(ctx, P.px(x), P.py(y), "#4263eb");
    ctx.fillStyle = textColor(); ctx.font = "12px ui-sans-serif";
    ctx.fillText("blue = current model   green dashed = least-squares solution", 8, 16);
  }, [pts, model, showSquares], W, H);

  const lossRef = useCanvas((ctx, w, h) => lineChart(ctx, [{ values: losses, color: "#e8590c" }], 30, 6, w - 36, h - 14, { title: "MSE during gradient descent", log: true }), [losses], 260, 160);

  return (
    <div className="widget">
      <h3>Least squares playground</h3>
      <p className="muted" style={{ marginTop: 0 }}>Click to add points (shift-click to remove the nearest). The orange squares are the squared errors - least squares makes their <b>total area</b> as small as possible.</p>
      <div className="controls">
        <button className="btn primary" onClick={() => setRunning(!running)}>{running ? "⏸ Pause GD" : "▶ Train with gradient descent"}</button>
        <button className="btn" onClick={() => { setRunning(false); setModel(best); }}>Solve with normal equation</button>
        <button className="btn" onClick={() => { setRunning(false); setModel({ w: 0, b: 5 }); setLosses([]); }}>Reset line</button>
        <button className="btn" onClick={() => { setPts(initialPoints(Math.floor(Math.random() * 1e6))); setLosses([]); }}>New data</button>
        <label><input type="checkbox" checked={showSquares} onChange={(e) => setShowSquares(e.target.checked)} /> squares</label>
        <label>η = {lr.toFixed(4)} <input type="range" min={-4} max={-1} step={0.05} value={Math.log10(lr)} onChange={(e) => setLr(10 ** Number(e.target.value))} /></label>
      </div>
      <div className="stat">ŷ = {fmt(model.w)}·x + {fmt(model.b)}   MSE = {fmt(mse(pts, model.w, model.b))}   |   optimum: w = {fmt(best.w)}, b = {fmt(best.b)}, MSE = {fmt(mse(pts, best.w, best.b))}</div>
      <div style={{ display: "flex", gap: 14, flexWrap: "wrap", alignItems: "flex-start", marginTop: 8 }}>
        <canvas ref={ref} style={{ cursor: "crosshair" }} onClick={(e) => {
          const [cx, cy] = canvasPos(e, W, H);
          const p: [number, number] = [P.dx(cx), P.dy(cy)];
          if (e.shiftKey && pts.length) {
            const i = pts.reduce((bi, q, j) => ((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 < (pts[bi][0] - p[0]) ** 2 + (pts[bi][1] - p[1]) ** 2 ? j : bi), 0);
            setPts(pts.filter((_, j) => j !== i));
          } else setPts([...pts, p]);
        }} />
        <div>
          <canvas ref={lossRef} />
          <p className="muted" style={{ maxWidth: 260, fontSize: "0.85rem" }}>Add one far-away point and watch the line swing: squared error punishes outliers heavily. Raise η until GD diverges.</p>
        </div>
      </div>
    </div>
  );
}
