"use client";
import { useMemo, useRef, useState } from "react";
import { fmt, rng, textColor, useCanvas, useLoop } from "./lib";

const LAYOUT = ["S...#....", ".##.#.##.", "....#..H.", ".#....#..", ".#.##.#.G", "...H....."];
const ACT = [[-1, 0], [0, 1], [1, 0], [0, -1]];
const ARROWS = "↑→↓←";
const R = LAYOUT.length, C = LAYOUT[0].length;
const cell = (r: number, c: number) => LAYOUT[r][c];
const terminal = (r: number, c: number) => "GH".includes(cell(r, c));
const reward = (r: number, c: number) => (cell(r, c) === "G" ? 1 : cell(r, c) === "H" ? -1 : -0.02);
const start = (() => { for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) if (cell(r, c) === "S") return [r, c]; return [0, 0]; })();
function move(r: number, c: number, a: number) {
  const nr = r + ACT[a][0], nc = c + ACT[a][1];
  return nr >= 0 && nr < R && nc >= 0 && nc < C && cell(nr, nc) !== "#" ? [nr, nc] : [r, c];
}

export default function GridWorld() {
  const [alpha, setAlpha] = useState(0.2);
  const [gamma, setGamma] = useState(0.95);
  const [eps, setEps] = useState(0.2);
  const [algo, setAlgo] = useState<"q" | "sarsa">("q");
  const [running, setRunning] = useState(false);
  const Q = useRef<number[][][]>(Array.from({ length: R }, () => Array.from({ length: C }, () => [0, 0, 0, 0])));
  const [episodes, setEpisodes] = useState(0);
  const [returns, setReturns] = useState<number[]>([]);
  const [agent, setAgent] = useState(start);
  const rand = useMemo(() => rng(7), []);
  const choose = (r: number, c: number) => (rand.next() < eps ? rand.int(4) : Q.current[r][c].indexOf(Math.max(...Q.current[r][c])));

  const runEpisode = () => {
    let [r, c] = start;
    let a = choose(r, c), total = 0;
    for (let t = 0; t < 200 && !terminal(r, c); t++) {
      const [nr, nc] = move(r, c, a);
      const rew = reward(nr, nc);
      const a2 = choose(nr, nc);
      const next = terminal(nr, nc) ? 0 : algo === "q" ? Math.max(...Q.current[nr][nc]) : Q.current[nr][nc][a2];
      Q.current[r][c][a] += alpha * (rew + gamma * next - Q.current[r][c][a]); // TD update
      total += rew; r = nr; c = nc; a = a2;
    }
    setEpisodes((e) => e + 1);
    setReturns((x) => [...x.slice(-199), total]);
  };
  useLoop(running, () => { for (let i = 0; i < 5; i++) runEpisode(); }, 20);

  const greedyPath = () => {
    const path = [start];
    let [r, c] = start;
    for (let t = 0; t < 60 && !terminal(r, c); t++) { [r, c] = move(r, c, Q.current[r][c].indexOf(Math.max(...Q.current[r][c]))); path.push([r, c]); }
    return path;
  };

  const S = 56;
  const ref = useCanvas((ctx) => {
    const path = greedyPath();
    for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
      const ch = cell(r, c), x = c * S, y = r * S;
      const v = Math.max(...Q.current[r][c]);
      ctx.fillStyle = ch === "#" ? "#495057" : ch === "G" ? "#2b8a3e" : ch === "H" ? "#c92a2a" : v >= 0 ? `rgba(43,138,62,${Math.min(0.6, v)})` : `rgba(201,42,42,${Math.min(0.6, -v)})`;
      ctx.fillRect(x + 1, y + 1, S - 2, S - 2);
      ctx.fillStyle = ch === "#" || ch === "G" || ch === "H" ? "#fff" : textColor();
      ctx.font = "bold 16px ui-sans-serif"; ctx.textAlign = "center";
      if (ch === "G") ctx.fillText("+1", x + S / 2, y + S / 2 + 6);
      else if (ch === "H") ctx.fillText("−1", x + S / 2, y + S / 2 + 6);
      else if (ch !== "#" && episodes > 0) {
        ctx.fillText(ARROWS[Q.current[r][c].indexOf(v)], x + S / 2, y + S / 2 + 2);
        ctx.font = "10px ui-monospace"; ctx.fillText(fmt(v, 2), x + S / 2, y + S - 6);
      }
      if (ch === "S") { ctx.font = "10px ui-sans-serif"; ctx.fillText("start", x + S / 2, y + 12); }
    }
    ctx.textAlign = "left";
    ctx.strokeStyle = "#f59f00"; ctx.lineWidth = 3; ctx.beginPath();
    path.forEach(([r, c], i) => (i ? ctx.lineTo(c * S + S / 2, r * S + S / 2) : ctx.moveTo(c * S + S / 2, r * S + S / 2)));
    ctx.stroke();
    ctx.fillStyle = "#f59f00"; ctx.beginPath(); ctx.arc(agent[1] * S + S / 2, agent[0] * S + S / 2, 7, 0, 7); ctx.fill();
  }, [episodes, agent], C * S, R * S);

  const avg = returns.length ? returns.slice(-50).reduce((a, b) => a + b, 0) / Math.min(50, returns.length) : 0;
  return (
    <div className="widget">
      <h3>Q-learning in a grid world</h3>
      <p className="muted" style={{ marginTop: 0 }}>The agent starts knowing nothing. Each step it updates <code>Q(s,a) ← Q(s,a) + α[r + γ·max Q(s′,·) − Q(s,a)]</code>. Cells show the best action and its value; the orange line is the current greedy route. Each step costs −0.02, the goal pays +1, holes −1.</p>
      <div className="controls">
        <label>algorithm <select value={algo} onChange={(e) => setAlgo(e.target.value as "q" | "sarsa")}><option value="q">Q-learning (off-policy)</option><option value="sarsa">SARSA (on-policy)</option></select></label>
        <label>α = {alpha.toFixed(2)} <input type="range" min={0.01} max={1} step={0.01} value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} /></label>
        <label>γ = {gamma.toFixed(2)} <input type="range" min={0.5} max={0.999} step={0.001} value={gamma} onChange={(e) => setGamma(Number(e.target.value))} /></label>
        <label>ε = {eps.toFixed(2)} <input type="range" min={0} max={1} step={0.01} value={eps} onChange={(e) => setEps(Number(e.target.value))} /></label>
      </div>
      <div className="controls">
        <button className="btn primary" onClick={() => setRunning(!running)}>{running ? "⏸ Pause" : "▶ Train"}</button>
        <button className="btn" onClick={runEpisode}>One episode</button>
        <button className="btn" onClick={() => { const p = greedyPath(); let i = 0; const id = setInterval(() => { setAgent(p[i++]); if (i >= p.length) clearInterval(id); }, 200); }}>Walk greedy policy</button>
        <button className="btn" onClick={() => { setRunning(false); Q.current = Array.from({ length: R }, () => Array.from({ length: C }, () => [0, 0, 0, 0])); setEpisodes(0); setReturns([]); setAgent(start); }}>Reset</button>
        <span className="stat">episodes {episodes} · avg return (last 50) {fmt(avg, 2)}</span>
      </div>
      <canvas ref={ref} />
      <p className="muted" style={{ fontSize: "0.85rem" }}>Set ε = 0 from the start: the agent never explores and can get stuck. With SARSA and high ε, the learned route keeps a safer distance from the holes, because SARSA learns the value of the exploring policy it actually follows.</p>
    </div>
  );
}
