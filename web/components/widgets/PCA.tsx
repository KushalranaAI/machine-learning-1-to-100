"use client";
import { useMemo, useState } from "react";
import { Plane, type Pt, dot, fmt, rng, textColor, useCanvas } from "./lib";

const W = 460, H = 420;
const P = new Plane(W, H, -3, 3, -3, 3);

export default function PCA() {
  const [angle, setAngle] = useState(30);
  const [sx, setSx] = useState(1.6);
  const [sy, setSy] = useState(0.45);
  const [showProj, setShowProj] = useState(true);
  const [seed, setSeed] = useState(1);
  const X = useMemo(() => {
    const r = rng(seed);
    const a = (angle * Math.PI) / 180;
    return Array.from({ length: 150 }, () => {
      const u = sx * r.normal(), v = sy * r.normal();
      return [u * Math.cos(a) - v * Math.sin(a), u * Math.sin(a) + v * Math.cos(a)] as Pt;
    });
  }, [angle, sx, sy, seed]);

  // Covariance matrix and its eigen-decomposition (closed form for 2×2 symmetric matrices).
  const pca = useMemo(() => {
    const n = X.length;
    const mx = X.reduce((s, p) => s + p[0], 0) / n, my = X.reduce((s, p) => s + p[1], 0) / n;
    let a = 0, b = 0, c = 0;
    for (const [x, y] of X) { a += (x - mx) ** 2; b += (x - mx) * (y - my); c += (y - my) ** 2; }
    a /= n - 1; b /= n - 1; c /= n - 1;
    const tr = a + c, det = a * c - b * b;
    const l1 = tr / 2 + Math.sqrt(Math.max((tr * tr) / 4 - det, 0)), l2 = tr - l1;
    let v1: Pt = Math.abs(b) > 1e-12 ? [l1 - c, b] : a >= c ? [1, 0] : [0, 1];
    const nrm = Math.hypot(...v1); v1 = [v1[0] / nrm, v1[1] / nrm];
    return { mean: [mx, my] as Pt, cov: [a, b, c], l1, l2, v1, v2: [-v1[1], v1[0]] as Pt };
  }, [X]);

  const ref = useCanvas((ctx) => {
    const { mean, l1, l2, v1, v2 } = pca;
    if (showProj) {
      ctx.strokeStyle = "rgba(232,89,12,0.35)"; ctx.lineWidth = 1;
      for (const p of X) {
        const t = (p[0] - mean[0]) * v1[0] + (p[1] - mean[1]) * v1[1];
        const q: Pt = [mean[0] + t * v1[0], mean[1] + t * v1[1]];
        ctx.beginPath(); ctx.moveTo(P.px(p[0]), P.py(p[1])); ctx.lineTo(P.px(q[0]), P.py(q[1])); ctx.stroke();
        dot(ctx, P.px(q[0]), P.py(q[1]), "#e8590c", 2.2, "transparent");
      }
    }
    for (const p of X) dot(ctx, P.px(p[0]), P.py(p[1]), "#4263eb", 3.2);
    const arrow = (v: Pt, len: number, color: string, label: string) => {
      const x0 = P.px(mean[0]), y0 = P.py(mean[1]);
      const x1 = P.px(mean[0] + v[0] * len), y1 = P.py(mean[1] + v[1] * len);
      ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      const ang = Math.atan2(y1 - y0, x1 - x0);
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1 - 10 * Math.cos(ang - 0.4), y1 - 10 * Math.sin(ang - 0.4)); ctx.lineTo(x1 - 10 * Math.cos(ang + 0.4), y1 - 10 * Math.sin(ang + 0.4)); ctx.fill();
      ctx.font = "bold 13px ui-sans-serif"; ctx.fillText(label, x1 + 6, y1 - 6);
    };
    arrow(v1, 1.5 * Math.sqrt(l1), "#2b8a3e", "PC1");
    arrow(v2, 1.5 * Math.sqrt(Math.max(l2, 0)), "#ae3ec9", "PC2");
    ctx.fillStyle = textColor(); ctx.font = "12px ui-sans-serif";
    ctx.fillText("arrows: eigenvectors of the covariance matrix, length = 1.5·√eigenvalue", 8, 16);
  }, [X, pca, showProj], W, H);

  const total = pca.l1 + pca.l2;
  return (
    <div className="widget">
      <h3>PCA: find the direction of maximum variance</h3>
      <div className="controls">
        <label>rotation {angle}° <input type="range" min={-90} max={90} value={angle} onChange={(e) => setAngle(Number(e.target.value))} /></label>
        <label>spread along main axis {sx.toFixed(2)} <input type="range" min={0.1} max={2.5} step={0.05} value={sx} onChange={(e) => setSx(Number(e.target.value))} /></label>
        <label>spread across {sy.toFixed(2)} <input type="range" min={0.05} max={2.5} step={0.05} value={sy} onChange={(e) => setSy(Number(e.target.value))} /></label>
        <label><input type="checkbox" checked={showProj} onChange={(e) => setShowProj(e.target.checked)} /> project onto PC1</label>
        <button className="btn" onClick={() => setSeed(seed + 1)}>Resample</button>
      </div>
      <div className="stat">Σ = [[{fmt(pca.cov[0])}, {fmt(pca.cov[1])}], [{fmt(pca.cov[1])}, {fmt(pca.cov[2])}]] · λ₁ = {fmt(pca.l1)}, λ₂ = {fmt(pca.l2)} · PC1 explains {fmt((100 * pca.l1) / total, 1)}% of the variance</div>
      <canvas ref={ref} style={{ marginTop: 8 }} />
      <p className="muted" style={{ fontSize: "0.85rem" }}>The orange points are the 1-D compression of the data. When “spread across” is small, almost nothing is lost - that is why PCA can drop dimensions. Make both spreads equal: the eigenvalues match and no direction is special.</p>
    </div>
  );
}
