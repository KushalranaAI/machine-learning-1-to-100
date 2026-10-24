"use client";
import { useMemo, useState } from "react";
import { CLASS_COLORS, Plane, type Pt, dot, makeDataset, textColor, useCanvas } from "./lib";

const W = 480, H = 420;
const P = new Plane(W, H, -1.3, 1.3, -1.3, 1.3);

function dbscan(X: Pt[], eps: number, minPts: number) {
  const nb = X.map((p) => X.map((q, j) => ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 <= eps * eps ? j : -1)).filter((j) => j >= 0));
  const core = nb.map((n) => n.length >= minPts);
  const label = new Array(X.length).fill(-1);
  let c = 0;
  for (let i = 0; i < X.length; i++) {
    if (label[i] !== -1 || !core[i]) continue;
    label[i] = c;
    const stack = [i];
    while (stack.length) {
      const p = stack.pop()!;
      if (!core[p]) continue;
      for (const q of nb[p]) if (label[q] === -1) { label[q] = c; stack.push(q); }
    }
    c++;
  }
  return { label, core, clusters: c };
}

export default function DBSCAN() {
  const [eps, setEps] = useState(0.15);
  const [minPts, setMinPts] = useState(5);
  const [dataName, setDataName] = useState("moons");
  const X = useMemo(() => {
    const d = makeDataset(dataName, 240, 0.07, 3);
    const r = makeDataset("blobs", 20, 0.6, 9).X; // sprinkle noise points
    return [...d.X, ...r];
  }, [dataName]);
  const res = useMemo(() => dbscan(X, eps, minPts), [X, eps, minPts]);
  const ref = useCanvas((ctx) => {
    X.forEach((p, i) => {
      if (res.core[i]) {
        ctx.beginPath(); ctx.arc(P.px(p[0]), P.py(p[1]), (eps / 2.6) * W, 0, 7);
        ctx.fillStyle = "rgba(120,120,120,0.05)"; ctx.fill();
      }
    });
    X.forEach((p, i) => {
      const l = res.label[i];
      const color = l < 0 ? "#868e96" : CLASS_COLORS[l % 8];
      dot(ctx, P.px(p[0]), P.py(p[1]), color, res.core[i] ? 4.5 : 3, l < 0 ? "transparent" : "rgba(0,0,0,0.5)");
      if (l < 0) { ctx.strokeStyle = textColor(); ctx.lineWidth = 1; ctx.strokeRect(P.px(p[0]) - 3, P.py(p[1]) - 3, 6, 6); }
    });
  }, [X, res, eps], W, H);
  const noise = res.label.filter((l) => l < 0).length;
  return (
    <div className="widget">
      <h3>DBSCAN: clusters from density</h3>
      <p className="muted" style={{ marginTop: 0 }}>A <b>core point</b> has at least minPts neighbours within ε (large dots, grey halo = its ε-ball). Clusters are chains of connected core points plus the <b>border points</b> they reach (small dots). Everything else is <b>noise</b> (grey squares).</p>
      <div className="controls">
        <label>Data <select value={dataName} onChange={(e) => setDataName(e.target.value)}>{["moons", "circles", "blobs", "spiral"].map((d) => <option key={d}>{d}</option>)}</select></label>
        <label>ε = {eps.toFixed(3)} <input type="range" min={0.03} max={0.4} step={0.005} value={eps} onChange={(e) => setEps(Number(e.target.value))} /></label>
        <label>minPts = {minPts} <input type="range" min={1} max={20} value={minPts} onChange={(e) => setMinPts(Number(e.target.value))} /></label>
        <span className="stat">{res.clusters} clusters · {noise} noise points</span>
      </div>
      <canvas ref={ref} />
      <p className="muted" style={{ fontSize: "0.85rem" }}>Unlike k-means, DBSCAN finds non-convex shapes (moons, circles) and doesn&apos;t need k - but a single ε can&apos;t handle clusters of very different densities.</p>
    </div>
  );
}
