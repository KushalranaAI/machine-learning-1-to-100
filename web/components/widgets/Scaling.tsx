"use client";
import { useMemo, useState } from "react";
import { Plane, type Pt, dot, fmt, rng, textColor, useCanvas } from "./lib";

type Method = "none" | "standard" | "minmax" | "robust";
const quantile = (a: number[], q: number) => { const s = [...a].sort((x, y) => x - y); const i = (s.length - 1) * q; const lo = Math.floor(i); return s[lo] + (s[Math.ceil(i)] - s[lo]) * (i - lo); };

function scale(X: Pt[], m: Method): Pt[] {
  if (m === "none") return X;
  const cols = [0, 1].map((j) => X.map((p) => p[j]));
  const f = cols.map((c) => {
    if (m === "standard") { const mu = c.reduce((a, b) => a + b, 0) / c.length; const sd = Math.sqrt(c.reduce((a, b) => a + (b - mu) ** 2, 0) / c.length); return (v: number) => (v - mu) / sd; }
    if (m === "minmax") { const lo = Math.min(...c), hi = Math.max(...c); return (v: number) => (v - lo) / (hi - lo); }
    const med = quantile(c, 0.5), iqr = quantile(c, 0.75) - quantile(c, 0.25); return (v: number) => (v - med) / iqr;
  });
  return X.map((p) => [f[0](p[0]), f[1](p[1])] as Pt);
}

export default function Scaling() {
  const [method, setMethod] = useState<Method>("none");
  const [outlier, setOutlier] = useState(false);
  const raw = useMemo(() => {
    const r = rng(3);
    // income in $ (huge scale) vs age in years (small scale)
    const pts: Pt[] = Array.from({ length: 120 }, () => [55000 + 15000 * r.normal(), 40 + 11 * r.normal()]);
    if (outlier) pts.push([400000, 45]);
    return pts;
  }, [outlier]);
  const X = useMemo(() => scale(raw, method), [raw, method]);
  const query = X[5];
  const nearest = useMemo(() => X.map((p, i) => [Math.hypot(p[0] - query[0], p[1] - query[1]), i]).sort((a, b) => a[0] - b[0]).slice(1, 6).map((d) => d[1]), [X, query]);

  const ref = useCanvas((ctx, w, h) => {
    const xs = X.map((p) => p[0]), ys = X.map((p) => p[1]);
    // Same scale on both axes so distances look like what a kNN model "sees".
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) * 0.6 || 1;
    const P = new Plane(w, h, cx - span, cx + span, cy - span * (h / w), cy + span * (h / w));
    X.forEach((p, i) => dot(ctx, P.px(p[0]), P.py(p[1]), i === 5 ? "#e8590c" : nearest.includes(i) ? "#2b8a3e" : "#4263eb", i === 5 ? 6 : 3.5));
    ctx.fillStyle = textColor(); ctx.font = "12px ui-sans-serif";
    ctx.fillText("x: income   y: age   (axes drawn at the SAME scale)", 8, 16);
    ctx.fillText("orange = query point, green = its 5 nearest neighbours", 8, 32);
  }, [X, nearest], 480, 380);

  const stats = [0, 1].map((j) => { const c = X.map((p) => p[j]); const mu = c.reduce((a, b) => a + b, 0) / c.length; return { min: Math.min(...c), max: Math.max(...c), mean: mu, sd: Math.sqrt(c.reduce((a, b) => a + (b - mu) ** 2, 0) / c.length) }; });
  return (
    <div className="widget">
      <h3>Why feature scaling matters</h3>
      <p className="muted" style={{ marginTop: 0 }}>Income is measured in tens of thousands, age in tens. Unscaled, Euclidean distance is <b>all income</b>: the nearest neighbours line up vertically, ignoring age completely.</p>
      <div className="controls">
        <label>scaler <select value={method} onChange={(e) => setMethod(e.target.value as Method)}>
          <option value="none">none (raw)</option><option value="standard">StandardScaler (z-score)</option><option value="minmax">MinMaxScaler [0, 1]</option><option value="robust">RobustScaler (median / IQR)</option>
        </select></label>
        <label><input type="checkbox" checked={outlier} onChange={(e) => setOutlier(e.target.checked)} /> add a billionaire outlier</label>
      </div>
      <table style={{ display: "table", fontSize: "0.85rem" }}>
        <thead><tr><th>feature</th><th>min</th><th>max</th><th>mean</th><th>std</th></tr></thead>
        <tbody>{["income", "age"].map((n, j) => <tr key={n}><td>{n}</td><td>{fmt(stats[j].min, 2)}</td><td>{fmt(stats[j].max, 2)}</td><td>{fmt(stats[j].mean, 2)}</td><td>{fmt(stats[j].sd, 2)}</td></tr>)}</tbody>
      </table>
      <canvas ref={ref} />
      <p className="muted" style={{ fontSize: "0.85rem" }}>With the outlier switched on, MinMax squashes everyone else into a sliver near 0; RobustScaler barely notices, because the median and IQR ignore extremes.</p>
    </div>
  );
}
