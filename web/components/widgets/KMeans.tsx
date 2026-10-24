"use client";
import { useMemo, useState } from "react";
import { CLASS_COLORS, CLASS_RGB, Plane, type Pt, blobs, canvasPos, dot, fmt, rng, textColor, useCanvas, useLoop } from "./lib";

const W = 480, H = 420;
const P = new Plane(W, H, -1.3, 1.3, -1.3, 1.3);
const d2 = (a: Pt, b: Pt) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;

function initCentroids(X: Pt[], k: number, method: string, seed: number): Pt[] {
  const r = rng(seed);
  if (method === "random") return Array.from({ length: k }, () => [...X[r.int(X.length)]] as Pt);
  // k-means++: next centre chosen with probability ∝ squared distance to the nearest existing centre
  const C: Pt[] = [[...X[r.int(X.length)]] as Pt];
  while (C.length < k) {
    const D = X.map((x) => Math.min(...C.map((c) => d2(x, c))));
    let u = r.next() * D.reduce((a, b) => a + b, 0);
    let i = 0;
    while (u > D[i] && i < X.length - 1) u -= D[i++];
    C.push([...X[i]] as Pt);
  }
  return C;
}

export default function KMeans() {
  const [seed, setSeed] = useState(4);
  const [trueK, setTrueK] = useState(4);
  const [k, setK] = useState(4);
  const [method, setMethod] = useState("random");
  const [extra, setExtra] = useState<Pt[]>([]);
  const X = useMemo(() => [...blobs(trueK, 240, 0.14, seed), ...extra], [trueK, seed, extra]);
  const [C, setC] = useState<Pt[]>(() => initCentroids(X, k, method, 1));
  const [phase, setPhase] = useState<"assign" | "update">("assign");
  const [labels, setLabels] = useState<number[] | null>(null);
  const [iter, setIter] = useState(0);
  const [running, setRunning] = useState(false);
  const [history, setHistory] = useState<number[]>([]);

  const restart = (initSeed = Math.floor(Math.random() * 1e6), kk = k, m = method, pts = X) => {
    setRunning(false); setC(initCentroids(pts, kk, m, initSeed)); setLabels(null); setPhase("assign"); setIter(0); setHistory([]);
  };

  const inertia = labels ? X.reduce((s, x, i) => s + d2(x, C[labels[i]]), 0) : NaN;

  const stepOnce = () => {
    if (phase === "assign") {
      const L = X.map((x) => C.reduce((best, c, j) => (d2(x, c) < d2(x, C[best]) ? j : best), 0));
      const converged = labels !== null && L.every((v, i) => v === labels[i]);
      setLabels(L); setPhase("update");
      setHistory((h) => [...h, X.reduce((s, x, i) => s + d2(x, C[L[i]]), 0)]);
      if (converged) setRunning(false);
    } else if (labels) {
      setC(C.map((c, j) => {
        const mine = X.filter((_, i) => labels[i] === j);
        return mine.length ? [mine.reduce((s, p) => s + p[0], 0) / mine.length, mine.reduce((s, p) => s + p[1], 0) / mine.length] : c;
      }));
      setPhase("assign"); setIter(iter + 1);
    }
  };
  useLoop(running, stepOnce, 2.5);

  const ref = useCanvas((ctx) => {
    const cell = 6;
    for (let i = 0; i < W; i += cell) for (let j = 0; j < H; j += cell) {
      const p: Pt = [P.dx(i), P.dy(j)];
      const c = C.reduce((b, q, m) => (d2(p, q) < d2(p, C[b]) ? m : b), 0);
      const [r, g, b] = CLASS_RGB[c % CLASS_RGB.length];
      ctx.fillStyle = `rgba(${r},${g},${b},0.10)`;
      ctx.fillRect(i, j, cell, cell);
    }
    X.forEach((x, i) => dot(ctx, P.px(x[0]), P.py(x[1]), labels ? CLASS_COLORS[labels[i] % 8] : "#adb5bd", 3.5));
    C.forEach((c, j) => {
      const x = P.px(c[0]), y = P.py(c[1]);
      ctx.fillStyle = CLASS_COLORS[j % 8]; ctx.strokeStyle = textColor(); ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(x - 9, y - 9); ctx.lineTo(x + 9, y + 9); ctx.moveTo(x + 9, y - 9); ctx.lineTo(x - 9, y + 9); ctx.stroke();
      ctx.beginPath(); ctx.arc(x, y, 6, 0, 7); ctx.fill(); ctx.stroke();
    });
  }, [X, C, labels], W, H);

  return (
    <div className="widget">
      <h3>k-means, one step at a time</h3>
      <p className="muted" style={{ marginTop: 0 }}>Lloyd&apos;s algorithm alternates two steps: <b>assign</b> each point to its nearest centroid (✕), then <b>update</b> each centroid to the mean of its points. Shaded regions are the Voronoi cells. Click to add points.</p>
      <div className="controls">
        <label>true clusters {trueK} <input type="range" min={1} max={7} value={trueK} onChange={(e) => { setTrueK(Number(e.target.value)); setExtra([]); }} /></label>
        <label>k = {k} <input type="range" min={1} max={8} value={k} onChange={(e) => { setK(Number(e.target.value)); restart(undefined, Number(e.target.value)); }} /></label>
        <label>init <select value={method} onChange={(e) => { setMethod(e.target.value); restart(undefined, k, e.target.value); }}><option value="random">random points</option><option value="kmeans++">k-means++</option></select></label>
        <button className="btn" onClick={() => { setSeed(seed + 1); setExtra([]); }}>New data</button>
      </div>
      <div className="controls">
        <button className="btn primary" onClick={stepOnce}>{phase === "assign" ? "① Assign points" : "② Move centroids"}</button>
        <button className="btn" onClick={() => setRunning(!running)}>{running ? "⏸ Pause" : "▶ Auto-run"}</button>
        <button className="btn" onClick={() => restart()}>Re-initialise</button>
        <span className="stat">iteration {iter} · inertia J = {Number.isFinite(inertia) ? fmt(inertia) : "-"}{history.length > 1 && history.at(-1) === history.at(-2) ? " · converged ✓" : ""}</span>
      </div>
      <canvas ref={ref} style={{ cursor: "crosshair" }} onClick={(e) => { const [x, y] = canvasPos(e, W, H); setExtra([...extra, [P.dx(x), P.dy(y)]]); }} />
      <p className="muted" style={{ fontSize: "0.85rem" }}>Try k-means with <b>random</b> init several times (Re-initialise): sometimes two centroids share one blob - a local minimum with higher inertia. k-means++ spreads the starting centroids out and avoids most of these.</p>
    </div>
  );
}
