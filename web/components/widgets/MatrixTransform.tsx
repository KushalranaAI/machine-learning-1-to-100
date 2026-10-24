"use client";
import { useState } from "react";
import { Plane, fmt, textColor, useCanvas } from "./lib";

const W = 440, H = 440;
const P = new Plane(W, H, -3, 3, -3, 3);

export default function MatrixTransform() {
  const [m, setM] = useState([1.5, 0.5, 0.5, 1]);
  const [a, b, c, d] = m;
  const det = a * d - b * c;
  const tr = a + d;
  const disc = (tr * tr) / 4 - det;
  const eig = disc >= 0 ? [tr / 2 + Math.sqrt(disc), tr / 2 - Math.sqrt(disc)] : null;
  const eigvec = (l: number): [number, number] => { const v: [number, number] = Math.abs(b) > 1e-9 ? [b, l - a] : Math.abs(c) > 1e-9 ? [l - d, c] : l === a ? [1, 0] : [0, 1]; const n = Math.hypot(...v); return [v[0] / n, v[1] / n]; };
  const T = (x: number, y: number): [number, number] => [a * x + b * y, c * x + d * y];

  const ref = useCanvas((ctx) => {
    ctx.strokeStyle = "rgba(128,128,128,0.25)"; ctx.lineWidth = 1;
    for (let k = -3; k <= 3; k++) { ctx.beginPath(); ctx.moveTo(P.px(k), 0); ctx.lineTo(P.px(k), H); ctx.moveTo(0, P.py(k)); ctx.lineTo(W, P.py(k)); ctx.stroke(); }
    ctx.strokeStyle = "rgba(66,99,235,0.55)";
    for (let k = -4; k <= 4; k++) {
      ctx.beginPath(); for (let t = -4; t <= 4; t += 0.5) { const [x, y] = T(k, t); t === -4 ? ctx.moveTo(P.px(x), P.py(y)) : ctx.lineTo(P.px(x), P.py(y)); } ctx.stroke();
      ctx.beginPath(); for (let t = -4; t <= 4; t += 0.5) { const [x, y] = T(t, k); t === -4 ? ctx.moveTo(P.px(x), P.py(y)) : ctx.lineTo(P.px(x), P.py(y)); } ctx.stroke();
    }
    ctx.strokeStyle = "#e8590c"; ctx.lineWidth = 2.5; ctx.beginPath();
    for (let i = 0; i <= 100; i++) { const th = (2 * Math.PI * i) / 100; const [x, y] = T(Math.cos(th), Math.sin(th)); i ? ctx.lineTo(P.px(x), P.py(y)) : ctx.moveTo(P.px(x), P.py(y)); }
    ctx.stroke();
    ctx.fillStyle = "rgba(232,89,12,0.12)";
    const sq = [T(0, 0), T(1, 0), T(1, 1), T(0, 1)];
    ctx.beginPath(); sq.forEach(([x, y], i) => (i ? ctx.lineTo(P.px(x), P.py(y)) : ctx.moveTo(P.px(x), P.py(y)))); ctx.fill();
    const arrow = ([x, y]: [number, number], color: string, label: string) => {
      ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(P.px(0), P.py(0)); ctx.lineTo(P.px(x), P.py(y)); ctx.stroke();
      ctx.font = "bold 12px ui-sans-serif"; ctx.fillText(label, P.px(x) + 5, P.py(y) - 5);
    };
    arrow(T(1, 0), "#2b8a3e", "A·e₁");
    arrow(T(0, 1), "#ae3ec9", "A·e₂");
    if (eig) eig.forEach((l, i) => { const v = eigvec(l); ctx.setLineDash([5, 4]); ctx.strokeStyle = textColor(); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(P.px(-3 * v[0]), P.py(-3 * v[1])); ctx.lineTo(P.px(3 * v[0]), P.py(3 * v[1])); ctx.stroke(); ctx.setLineDash([]); ctx.fillStyle = textColor(); ctx.font = "11px ui-sans-serif"; ctx.fillText(`eigvec ${i + 1} (λ=${l.toFixed(2)})`, P.px(2.2 * v[0]) + 4, P.py(2.2 * v[1])); });
  }, [m], W, H);

  const slider = (i: number, name: string) => (
    <label key={i}>{name} = {m[i].toFixed(2)} <input type="range" min={-2} max={2} step={0.05} value={m[i]} onChange={(e) => setM(m.map((v, j) => (j === i ? Number(e.target.value) : v)))} /></label>
  );
  const presets: Record<string, number[]> = { identity: [1, 0, 0, 1], scale: [2, 0, 0, 0.5], rotate45: [0.71, -0.71, 0.71, 0.71], shear: [1, 1, 0, 1], singular: [1, 2, 0.5, 1], reflect: [1, 0, 0, -1] };
  return (
    <div className="widget">
      <h3>A matrix is a transformation of space</h3>
      <p className="muted" style={{ marginTop: 0 }}>The blue grid and the orange unit circle are transformed by A = [[a, b], [c, d]]. Eigenvectors (dashed) are directions that only get stretched, by their eigenvalue λ.</p>
      <div className="controls">{slider(0, "a")}{slider(1, "b")}{slider(2, "c")}{slider(3, "d")}</div>
      <div className="controls">{Object.entries(presets).map(([k, v]) => <button key={k} className="btn" onClick={() => setM(v)}>{k}</button>)}</div>
      <div className="stat">det(A) = {fmt(det)} (area scaling of the orange square{Math.abs(det) < 1e-3 ? " - singular: space collapses onto a line, A has no inverse" : det < 0 ? ", negative: orientation flips" : ""}) · eigenvalues: {eig ? eig.map((l) => fmt(l)).join(", ") : "complex (a rotation)"}</div>
      <canvas ref={ref} style={{ marginTop: 8 }} />
    </div>
  );
}
