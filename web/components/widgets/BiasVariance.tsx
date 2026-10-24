"use client";
import { useMemo, useState } from "react";
import { Plane, dot, drawGrid, fmt, lineChart, rng, textColor, useCanvas } from "./lib";

const W = 520, H = 340;
const P = new Plane(W, H, -1.05, 1.05, -1.8, 1.8);
const truth = (x: number) => Math.sin(Math.PI * x);

/** Solve (AᵀA + λI) w = Aᵀy with Gaussian elimination (tiny ridge keeps high degrees stable). */
function polyFit(xs: number[], ys: number[], degree: number, lambda = 1e-6): number[] {
  const d = degree + 1;
  const M = Array.from({ length: d }, () => new Array(d + 1).fill(0));
  for (let k = 0; k < xs.length; k++) {
    const pw = Array.from({ length: d }, (_, i) => xs[k] ** i);
    for (let i = 0; i < d; i++) { for (let j = 0; j < d; j++) M[i][j] += pw[i] * pw[j]; M[i][d] += pw[i] * ys[k]; }
  }
  for (let i = 1; i < d; i++) M[i][i] += lambda;
  for (let c = 0; c < d; c++) {
    let piv = c;
    for (let r = c + 1; r < d; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    [M[c], M[piv]] = [M[piv], M[c]];
    for (let r = 0; r < d; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let k = c; k <= d; k++) M[r][k] -= f * M[c][k]; }
  }
  return M.map((row, i) => row[d] / row[i]);
}
const evalPoly = (w: number[], x: number) => w.reduce((s, c, i) => s + c * x ** i, 0);
const mse = (w: number[], xs: number[], ys: number[]) => xs.reduce((s, x, i) => s + (ys[i] - evalPoly(w, x)) ** 2, 0) / xs.length;

export default function BiasVariance() {
  const [seed, setSeed] = useState(7);
  const [degree, setDegree] = useState(3);
  const [noise, setNoise] = useState(0.3);
  const [nTrain, setNTrain] = useState(15);
  const data = useMemo(() => {
    const r = rng(seed);
    const make = (n: number) => { const xs = Array.from({ length: n }, () => 2 * r.next() - 1); return { xs, ys: xs.map((x) => truth(x) + noise * r.normal()) }; };
    return { train: make(nTrain), test: make(200) };
  }, [seed, noise, nTrain]);
  const curves = useMemo(() => {
    const tr: number[] = [], te: number[] = [];
    for (let d = 0; d <= 15; d++) { const w = polyFit(data.train.xs, data.train.ys, d); tr.push(mse(w, data.train.xs, data.train.ys)); te.push(mse(w, data.test.xs, data.test.ys)); }
    return { tr, te };
  }, [data]);
  const w = useMemo(() => polyFit(data.train.xs, data.train.ys, degree), [data, degree]);

  const ref = useCanvas((ctx) => {
    drawGrid(ctx, P, 0.5);
    const curve = (f: (x: number) => number, color: string, dash: number[] = []) => {
      ctx.setLineDash(dash); ctx.strokeStyle = color; ctx.lineWidth = 2.2; ctx.beginPath();
      for (let i = 0; i <= 300; i++) { const x = -1 + (2 * i) / 300; const y = Math.max(-3, Math.min(3, f(x))); i ? ctx.lineTo(P.px(x), P.py(y)) : ctx.moveTo(P.px(x), P.py(y)); }
      ctx.stroke(); ctx.setLineDash([]);
    };
    data.test.xs.forEach((x, i) => dot(ctx, P.px(x), P.py(data.test.ys[i]), "rgba(150,150,150,0.35)", 2.2, "transparent"));
    curve(truth, "rgba(43,138,62,0.9)", [6, 5]);
    curve((x) => evalPoly(w, x), "#e8590c");
    data.train.xs.forEach((x, i) => dot(ctx, P.px(x), P.py(data.train.ys[i]), "#4263eb"));
    ctx.fillStyle = textColor(); ctx.font = "12px ui-sans-serif";
    ctx.fillText(`degree ${degree} polynomial (orange) · true function (green) · grey = test points`, 8, 16);
  }, [data, w, degree], W, H);

  const chartRef = useCanvas((ctx, cw, ch) => {
    lineChart(ctx, [{ values: curves.tr, color: "#4263eb", label: "train MSE" }, { values: curves.te, color: "#e8590c", label: "test MSE" }], 30, 6, cw - 36, ch - 24, { log: true, title: "error vs degree" });
    const x = 30 + (degree / 15) * (cw - 36);
    ctx.strokeStyle = textColor(); ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(x, 6); ctx.lineTo(x, ch - 18); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = textColor(); ctx.font = "11px ui-sans-serif"; ctx.fillText("0", 28, ch - 6); ctx.fillText("15", cw - 18, ch - 6);
  }, [curves, degree], 280, 200);

  const verdict = degree <= 1 ? "Underfitting (high bias): too simple to follow the curve." : curves.te[degree] > 1.6 * Math.min(...curves.te) && degree > 5 ? "Overfitting (high variance): it chases the noise in the training points." : "Good balance - test error is near its minimum.";

  return (
    <div className="widget">
      <h3>Bias-variance trade-off with polynomial regression</h3>
      <div className="controls">
        <label>degree = {degree} <input type="range" min={0} max={15} value={degree} onChange={(e) => setDegree(Number(e.target.value))} /></label>
        <label>noise σ = {noise.toFixed(2)} <input type="range" min={0} max={0.8} step={0.05} value={noise} onChange={(e) => setNoise(Number(e.target.value))} /></label>
        <label>training points = {nTrain} <input type="range" min={6} max={80} value={nTrain} onChange={(e) => setNTrain(Number(e.target.value))} /></label>
        <button className="btn" onClick={() => setSeed(seed + 1)}>Resample data</button>
      </div>
      <div className="stat">train MSE = {fmt(curves.tr[degree])} · test MSE = {fmt(curves.te[degree])} · {verdict}</div>
      <div style={{ display: "flex", gap: 14, flexWrap: "wrap", alignItems: "flex-start", marginTop: 8 }}>
        <canvas ref={ref} />
        <div>
          <canvas ref={chartRef} />
          <p className="muted" style={{ maxWidth: 280, fontSize: "0.85rem" }}>Training error always falls as the degree grows; test error falls then rises - the U-shape. More training points push the overfitting point to higher degrees.</p>
        </div>
      </div>
    </div>
  );
}
