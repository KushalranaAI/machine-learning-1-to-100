// Shared helpers for the in-browser visualisers: seeded randomness, toy datasets and canvas plotting.
import { useEffect, useRef } from "react";
import { isDark } from "@/lib/theme";

export type Pt = [number, number];

/** Small seeded PRNG (mulberry32) so "resample" is reproducible. */
export function rng(seed = 1) {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const normal = () => {
    const u = Math.max(next(), 1e-12);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * next());
  };
  return { next, normal, int: (n: number) => Math.floor(next() * n) };
}

export type Dataset = { X: Pt[]; y: number[] };

/** 2-D toy datasets in roughly [-1, 1]². */
export function makeDataset(name: string, n = 160, noise = 0.15, seed = 1): Dataset {
  const r = rng(seed);
  const X: Pt[] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const c = i % 2;
    let p: Pt;
    if (name === "moons") {
      const t = Math.PI * r.next();
      p = c === 0 ? [Math.cos(t) - 0.5, Math.sin(t) - 0.25] : [0.5 - Math.cos(t), 0.25 - Math.sin(t)];
      p = [p[0] * 0.65, p[1] * 0.75];
    } else if (name === "circles") {
      const t = 2 * Math.PI * r.next();
      const rad = c === 0 ? 0.85 : 0.35;
      p = [rad * Math.cos(t), rad * Math.sin(t)];
    } else if (name === "xor") {
      const sx = r.next() < 0.5 ? -1 : 1;
      const sy = c === 0 ? sx : -sx;
      p = [sx * (0.2 + 0.6 * r.next()), sy * (0.2 + 0.6 * r.next())];
    } else if (name === "spiral") {
      const t = (i / n) * 3 * Math.PI + 0.3;
      const rad = (t / (3 * Math.PI)) * 0.9;
      const off = c * Math.PI;
      p = [rad * Math.cos(t + off), rad * Math.sin(t + off)];
    } else {
      p = c === 0 ? [-0.4, -0.3] : [0.4, 0.35];
    }
    X.push([p[0] + noise * r.normal(), p[1] + noise * r.normal()]);
    y.push(c);
  }
  return { X, y };
}

export function blobs(k: number, n: number, spread = 0.12, seed = 1): Pt[] {
  const r = rng(seed);
  const centers: Pt[] = Array.from({ length: k }, () => [r.next() * 1.5 - 0.75, r.next() * 1.5 - 0.75]);
  return Array.from({ length: n }, (_, i) => {
    const c = centers[i % k];
    return [c[0] + spread * r.normal(), c[1] + spread * r.normal()] as Pt;
  });
}

export const CLASS_COLORS = ["#4263eb", "#e8590c", "#2b8a3e", "#ae3ec9", "#f59f00", "#0c8599", "#c2255c", "#5c940d"];
export const CLASS_RGB: [number, number, number][] = [[66, 99, 235], [232, 89, 12], [43, 138, 62], [174, 62, 201], [245, 159, 0], [12, 133, 153], [194, 37, 92], [92, 148, 13]];

/** Maps data coordinates ↔ canvas pixels for a square-ish plotting box. */
export class Plane {
  constructor(public w: number, public h: number, public xmin = -1.2, public xmax = 1.2, public ymin = -1.2, public ymax = 1.2) {}
  px(x: number) { return ((x - this.xmin) / (this.xmax - this.xmin)) * this.w; }
  py(y: number) { return this.h - ((y - this.ymin) / (this.ymax - this.ymin)) * this.h; }
  dx(px: number) { return this.xmin + (px / this.w) * (this.xmax - this.xmin); }
  dy(py: number) { return this.ymin + ((this.h - py) / this.h) * (this.ymax - this.ymin); }
}

export function textColor() {
  return isDark() ? "#e5e9f0" : "#1d2330";
}

export function gridColor() {
  return isDark() ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.07)";
}

export function drawGrid(ctx: CanvasRenderingContext2D, P: Plane, step = 0.5) {
  ctx.strokeStyle = gridColor();
  ctx.lineWidth = 1;
  for (let x = Math.ceil(P.xmin / step) * step; x <= P.xmax; x += step) {
    ctx.beginPath(); ctx.moveTo(P.px(x), 0); ctx.lineTo(P.px(x), P.h); ctx.stroke();
  }
  for (let y = Math.ceil(P.ymin / step) * step; y <= P.ymax; y += step) {
    ctx.beginPath(); ctx.moveTo(0, P.py(y)); ctx.lineTo(P.w, P.py(y)); ctx.stroke();
  }
}

export function dot(ctx: CanvasRenderingContext2D, x: number, y: number, color: string, r = 4.5, stroke = "rgba(0,0,0,0.55)") {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 1;
  ctx.stroke();
}

/** Canvas with device-pixel-ratio scaling. Calls draw(ctx, w, h) whenever deps change. */
export function useCanvas(draw: (ctx: CanvasRenderingContext2D, w: number, h: number) => void, deps: unknown[], w = 520, h = 380) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = w * dpr;
    c.height = h * dpr;
    c.style.width = `${w}px`;
    c.style.aspectRatio = `${w} / ${h}`;
    const ctx = c.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    draw(ctx, w, h);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return ref;
}

/** Mouse position in canvas CSS pixels, robust to CSS scaling. */
export function canvasPos(e: React.MouseEvent<HTMLCanvasElement>, w: number, h: number): Pt {
  const r = e.currentTarget.getBoundingClientRect();
  return [((e.clientX - r.left) / r.width) * w, ((e.clientY - r.top) / r.height) * h];
}

/** requestAnimationFrame loop while `running` is true. */
export function useLoop(running: boolean, tick: () => void, fps = 30) {
  const tickRef = useRef(tick);
  tickRef.current = tick;
  useEffect(() => {
    if (!running) return;
    let id = 0;
    let last = 0;
    const loop = (t: number) => {
      if (t - last >= 1000 / fps) {
        last = t;
        tickRef.current();
      }
      id = requestAnimationFrame(loop);
    };
    id = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(id);
  }, [running, fps]);
}

/** Small line chart (e.g. a loss curve) drawn into a canvas context region. */
export function lineChart(ctx: CanvasRenderingContext2D, series: { values: number[]; color: string; label?: string }[], x0: number, y0: number, w: number, h: number, opts: { log?: boolean; title?: string } = {}) {
  const tc = textColor();
  ctx.strokeStyle = gridColor();
  ctx.strokeRect(x0, y0, w, h);
  const all = series.flatMap((s) => s.values).filter((v) => Number.isFinite(v) && (!opts.log || v > 0));
  if (!all.length) return;
  const f = (v: number) => (opts.log ? Math.log10(Math.max(v, 1e-12)) : v);
  let lo = Math.min(...all.map(f));
  let hi = Math.max(...all.map(f));
  if (hi - lo < 1e-9) { hi += 1; lo -= 1; }
  const n = Math.max(...series.map((s) => s.values.length));
  for (const s of series) {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    s.values.forEach((v, i) => {
      const x = x0 + (n > 1 ? (i / (n - 1)) * w : 0);
      const y = y0 + h - ((f(v) - lo) / (hi - lo)) * h;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  ctx.fillStyle = tc;
  ctx.font = "11px ui-sans-serif, system-ui";
  if (opts.title) ctx.fillText(opts.title, x0 + 6, y0 + 14);
  series.forEach((s, i) => {
    if (!s.label) return;
    ctx.fillStyle = s.color;
    ctx.fillText(s.label, x0 + w - 90, y0 + 14 + i * 13);
  });
}

export const fmt = (v: number, d = 3) => (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : v.toFixed(d));
