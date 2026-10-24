"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { CLASS_COLORS, Plane, type Pt, canvasPos, dot, fmt, lineChart, makeDataset, textColor, useCanvas, useLoop } from "./lib";
import { GaussianNB, KNN, LinearSVM, Logistic, MLP, type Model, Tree } from "./models";

export type ModelKind = "logistic" | "svm" | "knn" | "tree" | "nb" | "mlp";
const W = 440, H = 440;
const P = new Plane(W, H, -1.3, 1.3, -1.3, 1.3);
const ITERATIVE: ModelKind[] = ["logistic", "svm", "mlp"];
const NAMES: Record<ModelKind, string> = { logistic: "Logistic regression", svm: "Linear SVM (hinge loss)", knn: "k-nearest neighbours", tree: "Decision tree (CART)", nb: "Gaussian naive Bayes", mlp: "Neural network (MLP)" };

export default function Classifier2D({ initial = "logistic", allowSwitch = true, defaultData = "blobs" }: { initial?: ModelKind; allowSwitch?: boolean; defaultData?: string }) {
  const [kind, setKind] = useState<ModelKind>(initial);
  const [dataName, setDataName] = useState(defaultData);
  const [noise, setNoise] = useState(0.12);
  const [seed, setSeed] = useState(1);
  const [extra, setExtra] = useState<{ X: Pt[]; y: number[] }>({ X: [], y: [] });
  const [brush, setBrush] = useState(0);
  const [degree, setDegree] = useState(1);
  const [k, setK] = useState(5);
  const [depth, setDepth] = useState(3);
  const [C, setC] = useState(1);
  const [hidden, setHidden] = useState(8);
  const [running, setRunning] = useState(false);
  const [losses, setLosses] = useState<number[]>([]);
  const [tick, setTick] = useState(0);
  const modelRef = useRef<Model | null>(null);

  const data = useMemo(() => {
    const d = makeDataset(dataName, 160, noise, seed);
    return { X: [...d.X, ...extra.X], y: [...d.y, ...extra.y] };
  }, [dataName, noise, seed, extra]);

  // (Re)build the model whenever data or hyper-parameters change.
  useEffect(() => {
    setRunning(false);
    setLosses([]);
    if (kind === "logistic") modelRef.current = new Logistic(degree, 0.8);
    else if (kind === "svm") modelRef.current = new LinearSVM(C, 0.1, degree);
    else if (kind === "mlp") modelRef.current = new MLP([2, hidden, hidden, 1], 0.03);
    else if (kind === "knn") modelRef.current = new KNN(data.X, data.y, k);
    else if (kind === "tree") modelRef.current = new Tree(data.X, data.y, depth);
    else modelRef.current = new GaussianNB(data.X, data.y);
    setTick((t) => t + 1);
  }, [kind, data, degree, k, depth, C, hidden]);

  useLoop(running, () => {
    const m = modelRef.current;
    if (!m?.step) return;
    const loss = m.step(data.X, data.y, kind === "mlp" ? 3 : 10);
    setLosses((l) => [...l.slice(-399), loss]);
    setTick((t) => t + 1);
  }, 30);

  const accuracy = useMemo(() => {
    const m = modelRef.current;
    if (!m) return 0;
    return data.y.reduce((s, yi, i) => s + ((m.proba(data.X[i]) > 0.5 ? 1 : 0) === yi ? 1 : 0), 0) / data.y.length;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick, data]);

  const ref = useCanvas((ctx) => {
    const m = modelRef.current;
    if (m) {
      const cell = 6;
      for (let i = 0; i < W; i += cell) for (let j = 0; j < H; j += cell) {
        const p = m.proba([P.dx(i + cell / 2), P.dy(j + cell / 2)]);
        const a = Math.abs(p - 0.5) * 0.9;
        ctx.fillStyle = p > 0.5 ? `rgba(232,89,12,${a})` : `rgba(66,99,235,${a})`;
        ctx.fillRect(i, j, cell, cell);
      }
      if (m.splits) {
        ctx.strokeStyle = textColor(); ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
        for (const s of m.splits) { ctx.beginPath(); ctx.moveTo(P.px(s.x0), P.py(s.y0)); ctx.lineTo(P.px(s.x1), P.py(s.y1)); ctx.stroke(); }
        ctx.setLineDash([]);
      }
    }
    data.X.forEach((p, i) => dot(ctx, P.px(p[0]), P.py(p[1]), CLASS_COLORS[data.y[i]]));
  }, [tick, data], W, H);

  const lossRef = useCanvas((ctx, w, h) => lineChart(ctx, [{ values: losses, color: "#e8590c" }], 30, 6, w - 36, h - 14, { title: "training loss" }), [losses], 240, 140);

  return (
    <div className="widget">
      <h3>{allowSwitch ? "Classifier playground" : NAMES[kind]}</h3>
      <p className="muted" style={{ marginTop: 0 }}>Background colour = predicted P(orange). Click to add a point of the selected class.{ITERATIVE.includes(kind) ? " Press Train to watch gradient descent move the boundary." : " The model refits instantly."}</p>
      <div className="controls">
        {allowSwitch && (
          <label>Model <select value={kind} onChange={(e) => setKind(e.target.value as ModelKind)}>
            {Object.entries(NAMES).map(([key, name]) => <option key={key} value={key}>{name}</option>)}
          </select></label>
        )}
        <label>Data <select value={dataName} onChange={(e) => { setDataName(e.target.value); setExtra({ X: [], y: [] }); }}>
          {["blobs", "moons", "circles", "xor", "spiral"].map((d) => <option key={d}>{d}</option>)}
        </select></label>
        <label>noise {noise.toFixed(2)} <input type="range" min={0} max={0.4} step={0.01} value={noise} onChange={(e) => setNoise(Number(e.target.value))} /></label>
        <button className="btn" onClick={() => { setSeed(seed + 1); setExtra({ X: [], y: [] }); }}>Resample</button>
        <label>Add <select value={brush} onChange={(e) => setBrush(Number(e.target.value))}><option value={0}>blue</option><option value={1}>orange</option></select></label>
      </div>
      <div className="controls">
        {(kind === "logistic" || kind === "svm") && <label>polynomial features: degree {degree} <input type="range" min={1} max={5} value={degree} onChange={(e) => setDegree(Number(e.target.value))} /></label>}
        {kind === "svm" && <label>C = {C.toFixed(2)} <input type="range" min={-2} max={2} step={0.1} value={Math.log10(C)} onChange={(e) => setC(10 ** Number(e.target.value))} /></label>}
        {kind === "knn" && <label>k = {k} <input type="range" min={1} max={51} step={2} value={k} onChange={(e) => setK(Number(e.target.value))} /></label>}
        {kind === "tree" && <label>max depth = {depth} <input type="range" min={1} max={12} value={depth} onChange={(e) => setDepth(Number(e.target.value))} /></label>}
        {kind === "mlp" && <label>hidden units per layer = {hidden} <input type="range" min={2} max={16} value={hidden} onChange={(e) => setHidden(Number(e.target.value))} /></label>}
        {ITERATIVE.includes(kind) && <button className="btn primary" onClick={() => setRunning(!running)}>{running ? "⏸ Pause" : "▶ Train"}</button>}
        <span className="stat">training accuracy = {fmt(100 * accuracy, 1)}% · {data.X.length} points</span>
      </div>
      <div style={{ display: "flex", gap: 14, flexWrap: "wrap", alignItems: "flex-start" }}>
        <canvas ref={ref} style={{ cursor: "crosshair" }} onClick={(e) => { const [x, y] = canvasPos(e, W, H); setExtra({ X: [...extra.X, [P.dx(x), P.dy(y)]], y: [...extra.y, brush] }); }} />
        {ITERATIVE.includes(kind) && <canvas ref={lossRef} />}
      </div>
    </div>
  );
}
