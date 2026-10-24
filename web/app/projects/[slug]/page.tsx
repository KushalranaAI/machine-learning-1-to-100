import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import Markdown from "@/components/Markdown";
import Mermaid from "@/components/Mermaid";
import ProjectSteps from "@/components/ProjectSteps";
import MarkComplete from "@/components/progress/MarkComplete";
import Icon from "@/components/ui/Icon";
import { flatLessons, getProject, getProjects } from "@/lib/content";

export function generateStaticParams() {
  return getProjects().map((p) => ({ slug: p.slug }));
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const { slug } = await params;
  return { title: getProject(slug)?.title ?? "Project" };
}

export default async function ProjectPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const p = getProject(slug);
  if (!p) notFound();
  const lessons = flatLessons();
  return (
    <article>
      <header className="page-head">
        <div className="eyebrow"><Link href="/projects"><Icon name="rocket" size={14} /> Projects</Link> · Project {p.order}</div>
        <h1>{p.title}</h1>
        <p className="lead">{p.summary}</p>
        <div className="meta-row">
          <span className={`pill ${p.level.toLowerCase()}`}>{p.level}</span>
          {p.algorithms.map((a) => <span key={a} className="pill accent">{a}</span>)}
          <span className="spacer" />
          <MarkComplete id={`project/${slug}`} noun="project" />
        </div>
      </header>

      <div className="info-grid">
        <section className="card overview-card wide"><h4><Icon name="target" size={16} /> The problem</h4><Markdown>{p.problem}</Markdown></section>
        <section className="card overview-card">
          <h4><Icon name="database" size={16} /> Dataset</h4>
          <p style={{ margin: "0 0 6px" }}><b>{p.dataset.name}</b>{p.dataset.size ? ` · ${p.dataset.size}` : ""}</p>
          <p style={{ margin: "0 0 6px" }}>Source: <a href={p.dataset.url} target="_blank" rel="noreferrer">{p.dataset.source}</a></p>
          {p.dataset.kaggle && <p style={{ margin: 0 }}>Kaggle: <code>{p.dataset.kaggle}</code></p>}
          {p.dataset.license && <p className="muted small" style={{ margin: "6px 0 0" }}>{p.dataset.license}</p>}
        </section>
        <section className="card overview-card"><h4><Icon name="sliders" size={16} /> Skills you practise</h4><ul>{p.skills.map((s) => <li key={s}>{s}</li>)}</ul></section>
        <section className="card overview-card">
          <h4><Icon name="book" size={16} /> Lessons this builds on</h4>
          <ul>
            {p.lessons.map((id) => {
              const [s, t] = id.split("/");
              const l = lessons.find((x) => x.section === s && x.slug === t);
              return <li key={id}><Link href={`/learn/${id}`}>{l?.title ?? id}</Link></li>;
            })}
          </ul>
        </section>
      </div>

      {p.workflow && (
        <>
          <h2 className="section-title">Project workflow</h2>
          <Mermaid chart={p.workflow} />
        </>
      )}

      <ProjectSteps slug={slug} title={p.title} dir={p.dir.replace(/\\/g, "/")} steps={p.steps} problem={p.problem} body={p.body} />

      <h2 className="section-title">Full write-up</h2>
      <Markdown>{p.body}</Markdown>
    </article>
  );
}
