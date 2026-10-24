import Link from "next/link";
import { getProjects } from "@/lib/content";

export default function ProjectsPage() {
  const projects = getProjects();
  return (
    <>
      <h1>🚀 Projects</h1>
      <p className="lead">Five end-to-end projects, from a first regression to a neural network built from scratch - each on a real dataset.</p>
      {projects.map((p) => (
        <Link key={p.slug} href={`/projects/${p.slug}`} className="card" style={{ display: "block", color: "inherit", textDecoration: "none", marginBottom: 14 }}>
          <span className={`pill ${p.level.toLowerCase()}`}>{p.level}</span>
          <span className="pill">{p.dataset.name}</span>
          <h3 style={{ margin: "4px 0 6px" }}>{p.order}. {p.title}</h3>
          <p className="muted" style={{ margin: "0 0 8px" }}>{p.summary}</p>
          {p.algorithms.map((a) => <span key={a} className="pill accent">{a}</span>)}
        </Link>
      ))}
    </>
  );
}
