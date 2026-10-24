import Link from "next/link";
import { notFound } from "next/navigation";
import Markdown from "@/components/Markdown";
import Mermaid from "@/components/Mermaid";
import RunPanel from "@/components/RunPanel";
import { flatLessons, getProject } from "@/lib/content";

export default async function ProjectPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const p = getProject(slug);
  if (!p) notFound();
  const lessons = flatLessons();
  return (
    <>
      <div className="stat"><Link href="/projects">🚀 Projects</Link> · Project {p.order}</div>
      <h1>{p.title}</h1>
      <p className="lead">{p.summary}</p>
      <span className={`pill ${p.level.toLowerCase()}`}>{p.level}</span>
      {p.algorithms.map((a) => <span key={a} className="pill accent">{a}</span>)}

      <div className="info-grid">
        <div className="card"><h4>The problem</h4><Markdown>{p.problem}</Markdown></div>
        <div className="card">
          <h4>Dataset</h4>
          <p style={{ margin: "0 0 6px" }}><b>{p.dataset.name}</b>{p.dataset.size ? ` · ${p.dataset.size}` : ""}</p>
          <p style={{ margin: "0 0 6px" }}>Source: <a href={p.dataset.url} target="_blank" rel="noreferrer">{p.dataset.source}</a></p>
          {p.dataset.kaggle && <p style={{ margin: 0 }}>Kaggle: <code>{p.dataset.kaggle}</code></p>}
          {p.dataset.license && <p className="muted" style={{ margin: "6px 0 0", fontSize: "0.85rem" }}>{p.dataset.license}</p>}
        </div>
        <div className="card"><h4>Skills you practise</h4><ul>{p.skills.map((s) => <li key={s}>{s}</li>)}</ul></div>
        <div className="card">
          <h4>Lessons this builds on</h4>
          <ul>
            {p.lessons.map((id) => {
              const [s, t] = id.split("/");
              const l = lessons.find((x) => x.section === s && x.slug === t);
              return <li key={id}><Link href={`/learn/${id}`}>{l?.title ?? id}</Link></li>;
            })}
          </ul>
        </div>
      </div>

      {p.workflow && (<><h2>Project workflow</h2><Mermaid chart={p.workflow} /></>)}

      <h2>Step-by-step guide</h2>
      <ol className="steps">
        {p.steps.map((step) => (
          <li key={step.title}>
            <h3 style={{ marginTop: 0 }}>{step.title}</h3>
            <Markdown>{step.description}</Markdown>
            {step.script && <RunPanel script={`${p.dir}/${step.script.split(" ")[0]}`} args={step.script.split(" ").slice(1)} label="Run this step" />}
          </li>
        ))}
      </ol>

      <h2>Full write-up</h2>
      <Markdown>{p.body}</Markdown>
    </>
  );
}
