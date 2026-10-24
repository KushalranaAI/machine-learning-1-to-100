"use client";
import { useProgress } from "./progress";

export default function ProgressSummary({ ids }: { ids: string[] }) {
  const p = useProgress();
  const done = ids.filter((id) => p[id]).length;
  const pct = ids.length ? Math.round((100 * done) / ids.length) : 0;
  return (
    <div style={{ maxWidth: 420 }}>
      <div className="stat" style={{ marginBottom: 6 }}>Your progress: {done} / {ids.length} lessons ({pct}%)</div>
      <div className="progress-bar"><div style={{ width: `${pct}%` }} /></div>
    </div>
  );
}
