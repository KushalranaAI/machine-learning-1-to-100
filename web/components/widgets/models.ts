// Tiny in-browser versions of the course's classifiers (binary, 2-D), mirroring mlscratch/.
import type { Pt } from "./lib";

export interface Model {
  /** P(class 1 | x) */
  proba(p: Pt): number;
  /** iterative models train in steps so the boundary can be animated */
  step?(X: Pt[], y: number[], n: number): number; // returns current loss
  splits?: { x0: number; y0: number; x1: number; y1: number }[];
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, z))));

/** Polynomial feature map up to `degree` (degree 1 = linear model). */
export function features(p: Pt, degree: number): number[] {
  const out: number[] = [];
  for (let d = 1; d <= degree; d++) for (let i = 0; i <= d; i++) out.push(p[0] ** (d - i) * p[1] ** i);
  return out;
}

export class Logistic implements Model {
  w: number[];
  b = 0;
  constructor(public degree = 1, public lr = 0.5, public l2 = 0.0) {
    this.w = new Array(features([0, 0], degree).length).fill(0);
  }
  score(p: Pt) { return features(p, this.degree).reduce((s, f, i) => s + f * this.w[i], this.b); }
  proba(p: Pt) { return sigmoid(this.score(p)); }
  step(X: Pt[], y: number[], n: number) {
    let loss = 0;
    for (let it = 0; it < n; it++) {
      const gw = new Array(this.w.length).fill(0);
      let gb = 0;
      loss = 0;
      X.forEach((p, k) => {
        const f = features(p, this.degree);
        const q = sigmoid(f.reduce((s, v, i) => s + v * this.w[i], this.b));
        const e = q - y[k]; // ∂(cross-entropy)/∂z
        f.forEach((v, i) => (gw[i] += e * v));
        gb += e;
        loss -= y[k] * Math.log(q + 1e-12) + (1 - y[k]) * Math.log(1 - q + 1e-12);
      });
      const m = X.length;
      this.w = this.w.map((w, i) => w - this.lr * (gw[i] / m + this.l2 * w));
      this.b -= (this.lr * gb) / m;
      loss /= m;
    }
    return loss;
  }
}

export class LinearSVM implements Model {
  w = [0, 0];
  b = 0;
  constructor(public C = 1, public lr = 0.05, public degree = 1) {
    this.w = new Array(features([0, 0], degree).length).fill(0);
  }
  score(p: Pt) { return features(p, this.degree).reduce((s, f, i) => s + f * this.w[i], this.b); }
  proba(p: Pt) { return sigmoid(3 * this.score(p)); }
  step(X: Pt[], y: number[], n: number) {
    let loss = 0;
    for (let it = 0; it < n; it++) {
      // subgradient of ½||w||² + C·mean(max(0, 1 - t·f(x)))
      const gw = this.w.map((w) => w);
      let gb = 0;
      loss = 0.5 * this.w.reduce((s, w) => s + w * w, 0);
      X.forEach((p, k) => {
        const t = y[k] === 1 ? 1 : -1;
        const f = features(p, this.degree);
        const margin = t * (f.reduce((s, v, i) => s + v * this.w[i], this.b));
        if (margin < 1) {
          f.forEach((v, i) => (gw[i] -= (this.C * t * v) / X.length));
          gb -= (this.C * t) / X.length;
          loss += (this.C * (1 - margin)) / X.length;
        }
      });
      this.w = this.w.map((w, i) => w - this.lr * gw[i]);
      this.b -= this.lr * gb;
    }
    return loss;
  }
}

export class KNN implements Model {
  constructor(public X: Pt[], public y: number[], public k = 5) {}
  proba(p: Pt) {
    const d = this.X.map((q, i) => [(q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2, this.y[i]]);
    d.sort((a, b) => a[0] - b[0]);
    const top = d.slice(0, this.k);
    return top.reduce((s, v) => s + v[1], 0) / Math.max(top.length, 1);
  }
}

export class GaussianNB implements Model {
  stats: { prior: number; mu: number[]; v: number[] }[] = [];
  constructor(X: Pt[], y: number[]) {
    for (const c of [0, 1]) {
      const pts = X.filter((_, i) => y[i] === c);
      const n = Math.max(pts.length, 1);
      const mu = [0, 1].map((j) => pts.reduce((s, p) => s + p[j], 0) / n);
      const v = [0, 1].map((j) => pts.reduce((s, p) => s + (p[j] - mu[j]) ** 2, 0) / n + 1e-3);
      this.stats.push({ prior: pts.length / X.length, mu, v });
    }
  }
  logp(p: Pt, c: number) {
    const s = this.stats[c];
    return Math.log(s.prior + 1e-12) + [0, 1].reduce((acc, j) => acc - 0.5 * Math.log(2 * Math.PI * s.v[j]) - (p[j] - s.mu[j]) ** 2 / (2 * s.v[j]), 0);
  }
  proba(p: Pt) {
    const a = this.logp(p, 0), b = this.logp(p, 1);
    return 1 / (1 + Math.exp(a - b));
  }
}

type TNode = { leaf: true; p: number } | { leaf: false; f: number; t: number; l: TNode; r: TNode };

const gini = (n1: number, n: number) => (n ? 1 - (n1 / n) ** 2 - (1 - n1 / n) ** 2 : 0);

/** CART with Gini impurity; records split lines so the widget can draw them. */
export class Tree implements Model {
  root: TNode;
  splits: { x0: number; y0: number; x1: number; y1: number }[] = [];
  constructor(X: Pt[], y: number[], public maxDepth = 3, public minLeaf = 1) {
    this.root = this.grow(X.map((_, i) => i), X, y, 0, [-1.3, 1.3, -1.3, 1.3]);
  }
  grow(idx: number[], X: Pt[], y: number[], depth: number, box: number[]): TNode {
    const n1 = idx.reduce((s, i) => s + y[i], 0);
    const node: TNode = { leaf: true, p: idx.length ? n1 / idx.length : 0.5 };
    if (depth >= this.maxDepth || n1 === 0 || n1 === idx.length || idx.length < 2 * this.minLeaf) return node;
    let best = { gain: 1e-9, f: -1, t: 0 };
    const parent = gini(n1, idx.length);
    for (const f of [0, 1]) {
      const sorted = [...idx].sort((a, b) => X[a][f] - X[b][f]);
      let left1 = 0;
      for (let k = 0; k < sorted.length - 1; k++) {
        left1 += y[sorted[k]];
        const nl = k + 1, nr = sorted.length - nl;
        if (nl < this.minLeaf || nr < this.minLeaf || X[sorted[k]][f] === X[sorted[k + 1]][f]) continue;
        const g = parent - (nl * gini(left1, nl) + nr * gini(n1 - left1, nr)) / sorted.length;
        if (g > best.gain) best = { gain: g, f, t: (X[sorted[k]][f] + X[sorted[k + 1]][f]) / 2 };
      }
    }
    if (best.f < 0) return node;
    const [x0, x1, y0, y1] = box;
    this.splits.push(best.f === 0 ? { x0: best.t, y0, x1: best.t, y1 } : { x0, y0: best.t, x1, y1: best.t });
    const L = idx.filter((i) => X[i][best.f] <= best.t), R = idx.filter((i) => X[i][best.f] > best.t);
    const lb = best.f === 0 ? [x0, best.t, y0, y1] : [x0, x1, y0, best.t];
    const rb = best.f === 0 ? [best.t, x1, y0, y1] : [x0, x1, best.t, y1];
    return { leaf: false, f: best.f, t: best.t, l: this.grow(L, X, y, depth + 1, lb), r: this.grow(R, X, y, depth + 1, rb) };
  }
  proba(p: Pt) {
    let n = this.root;
    while (!n.leaf) n = p[n.f] <= n.t ? n.l : n.r;
    return n.p;
  }
}

/** 2 → H → H → 1 tanh network trained with Adam on cross-entropy (backprop written out). */
export class MLP implements Model {
  W: number[][][];
  b: number[][];
  m: number[][][];
  v: number[][][];
  mb: number[][];
  vb: number[][];
  t = 0;
  constructor(public sizes = [2, 8, 8, 1], public lr = 0.03, seed = 1) {
    let s = seed;
    const rnd = () => { s = (s * 16807) % 2147483647; return s / 2147483647 - 0.5; };
    this.W = sizes.slice(1).map((n, l) => Array.from({ length: sizes[l] }, () => Array.from({ length: n }, () => rnd() * 2 * Math.sqrt(1 / sizes[l]) * 1.7)));
    this.b = sizes.slice(1).map((n) => new Array(n).fill(0));
    const zeros = () => this.W.map((L) => L.map((r) => r.map(() => 0)));
    this.m = zeros(); this.v = zeros();
    this.mb = this.b.map((r) => r.map(() => 0)); this.vb = this.b.map((r) => r.map(() => 0));
  }
  forward(p: Pt) {
    const acts: number[][] = [[p[0], p[1]]];
    this.W.forEach((L, l) => {
      const a = acts[acts.length - 1];
      const z = this.b[l].map((bj, j) => a.reduce((s, ai, i) => s + ai * L[i][j], bj));
      acts.push(l === this.W.length - 1 ? z : z.map(Math.tanh));
    });
    return acts;
  }
  proba(p: Pt) { return sigmoid(this.forward(p).at(-1)![0]); }
  step(X: Pt[], y: number[], n: number) {
    let loss = 0;
    for (let it = 0; it < n; it++) {
      const gW = this.W.map((L) => L.map((r) => r.map(() => 0)));
      const gb = this.b.map((r) => r.map(() => 0));
      loss = 0;
      X.forEach((p, k) => {
        const acts = this.forward(p);
        const q = sigmoid(acts.at(-1)![0]);
        loss -= y[k] * Math.log(q + 1e-12) + (1 - y[k]) * Math.log(1 - q + 1e-12);
        let delta = [q - y[k]]; // ∂L/∂z at the output
        for (let l = this.W.length - 1; l >= 0; l--) {
          const a = acts[l];
          delta.forEach((dj, j) => { gb[l][j] += dj; a.forEach((ai, i) => (gW[l][i][j] += ai * dj)); });
          if (l > 0) delta = a.map((ai, i) => (1 - ai * ai) * delta.reduce((s, dj, j) => s + dj * this.W[l][i][j], 0));
        }
      });
      this.t++;
      const m = X.length, b1 = 0.9, b2 = 0.999;
      const upd = (w: number, g: number, mo: number, ve: number): [number, number, number] => {
        g /= m;
        const nm = b1 * mo + (1 - b1) * g, nv = b2 * ve + (1 - b2) * g * g;
        return [w - (this.lr * (nm / (1 - b1 ** this.t))) / (Math.sqrt(nv / (1 - b2 ** this.t)) + 1e-8), nm, nv];
      };
      this.W.forEach((L, l) => L.forEach((r, i) => r.forEach((w, j) => { [L[i][j], this.m[l][i][j], this.v[l][i][j]] = upd(w, gW[l][i][j], this.m[l][i][j], this.v[l][i][j]); })));
      this.b.forEach((r, l) => r.forEach((w, j) => { [r[j], this.mb[l][j], this.vb[l][j]] = upd(w, gb[l][j], this.mb[l][j], this.vb[l][j]); }));
      loss /= m;
    }
    return loss;
  }
}
