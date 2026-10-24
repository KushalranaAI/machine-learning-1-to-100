import Link from "next/link";
import Mermaid from "@/components/Mermaid";
import ProgressSummary from "@/components/ProgressSummary";
import { getProjects, getSections } from "@/lib/content";

const PIPELINE = `flowchart LR
  A[Collect data] --> B[Wrangle & clean] --> C[Explore EDA] --> D[Engineer features]
  D --> E[Select / extract features] --> F[Train models] --> G[Evaluate & tune]
  G -->|not good enough| D
  G --> H[Deploy & monitor]`;

export default function Home() {
  const sections = getSections();
  const projects = getProjects();
  const ids = sections.flatMap((s) => s.lessons.map((l) => `${s.slug}/${l.slug}`));
  return (
    <>
      <h1>Machine Learning 1 → 100</h1>
      <p className="lead">
        Every classical ML algorithm, built from its math in plain Python/NumPy (and C++), explained with background,
        workflows, interactive visualisations and runnable code - then used in five real projects.
      </p>
      <ProgressSummary ids={ids} />

      <h2>The machine-learning workflow you will master</h2>
      <Mermaid chart={PIPELINE} />

      <h2>Roadmap</h2>
      <div className="grid">
        {sections.map((s) => (
          <Link key={s.slug} href={`/learn/${s.slug}`} className="card" style={{ color: "inherit", textDecoration: "none" }}>
            <div style={{ fontSize: "1.6rem" }}>{s.icon}</div>
            <div className="stat">Part {s.order}</div>
            <h3 style={{ margin: "4px 0 6px" }}>{s.title}</h3>
            <p className="muted" style={{ margin: "0 0 8px", fontSize: "0.9rem" }}>{s.description}</p>
            <span className="pill">{s.lessons.length} lessons</span>
          </Link>
        ))}
      </div>

      <h2>Projects</h2>
      <div className="grid">
        {projects.map((p) => (
          <Link key={p.slug} href={`/projects/${p.slug}`} className="card" style={{ color: "inherit", textDecoration: "none" }}>
            <span className={`pill ${p.level.toLowerCase()}`}>{p.level}</span>
            <h3 style={{ margin: "4px 0 6px" }}>{p.title}</h3>
            <p className="muted" style={{ margin: 0, fontSize: "0.9rem" }}>{p.summary}</p>
          </Link>
        ))}
      </div>

      <h2>How to learn with this app</h2>
      <ol>
        <li><b>Learn</b> - read the background, intuition and the full derivation.</li>
        <li><b>Workflow</b> - see where the step sits in a real ML pipeline.</li>
        <li><b>Interactive</b> - drag points, move sliders and step through the algorithm.</li>
        <li><b>From-scratch code</b> - read the exact implementation in <code>mlscratch/</code> (and <code>cpp/</code>).</li>
        <li><b>Run demo</b> - execute the lesson script on your machine and inspect its plots.</li>
        <li><b>Build</b> - apply everything in the projects with real datasets from Kaggle and the web.</li>
      </ol>
    </>
  );
}
