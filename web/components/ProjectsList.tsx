"use client";
import Link from "next/link";
import type { NavProject } from "@/lib/nav";
import { projectId, stepId } from "@/lib/nav";
import { useProgress } from "@/lib/progress";
import Icon from "./ui/Icon";

export default function ProjectsList({ projects }: { projects: (NavProject & { algorithms: string[] })[] }) {
  const p = useProgress();
  return (
    <div className="grid projects-grid">
      {projects.map((pr, n) => {
        const isDone = !!p.done[projectId(pr.slug)];
        const stepsDone = Array.from({ length: pr.steps }).filter((_, i) => p.done[stepId(pr.slug, i)]).length;
        return (
          <Link key={pr.slug} href={`/projects/${pr.slug}`} className="card link-card lift">
            <span className="card-foot" style={{ marginTop: 0 }}>
              <span className="muted small">Project {n + 1}</span>
              <span className={`pill ${pr.level.toLowerCase()}`}>{pr.level}</span>
              {isDone && <span className="pill good"><Icon name="check" size={12} /> done</span>}
            </span>
            <b className="card-title">{pr.title}</b>
            <span className="muted small clamp">{pr.summary}</span>
            <span className="card-foot">
              <span className="pill"><Icon name="database" size={12} /> {pr.dataset}</span>
              {pr.algorithms.slice(0, 3).map((a) => <span key={a} className="pill accent">{a}</span>)}
            </span>
            <div className="bar" aria-label={`${stepsDone} of ${pr.steps} steps`}><div style={{ width: `${pr.steps ? (100 * (isDone ? pr.steps : stepsDone)) / pr.steps : 0}%` }} /></div>
          </Link>
        );
      })}
    </div>
  );
}
