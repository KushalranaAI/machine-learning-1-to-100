import Link from "next/link";
import { notFound } from "next/navigation";
import Markdown from "@/components/Markdown";
import { getSectionBody, getSections } from "@/lib/content";

export default async function SectionPage({ params }: { params: Promise<{ section: string }> }) {
  const { section } = await params;
  const s = getSections().find((x) => x.slug === section);
  if (!s) notFound();
  return (
    <>
      <div className="stat">Part {s.order}</div>
      <h1>{s.icon} {s.title}</h1>
      <p className="lead">{s.description}</p>
      <Markdown>{getSectionBody(section)}</Markdown>
      <h2>Lessons</h2>
      <div className="grid">
        {s.lessons.map((l, i) => (
          <Link key={l.slug} href={`/learn/${s.slug}/${l.slug}`} className="card" style={{ color: "inherit", textDecoration: "none" }}>
            <div className="stat">{s.order}.{i + 1} · {l.minutes} min</div>
            <h3 style={{ margin: "4px 0 6px" }}>{l.title}</h3>
            <p className="muted" style={{ margin: "0 0 8px", fontSize: "0.9rem" }}>{l.summary}</p>
            <span className={`pill ${l.level}`}>{l.level}</span>
            {l.widgets.length > 0 && <span className="pill accent">interactive</span>}
            {l.demo && <span className="pill">runnable demo</span>}
          </Link>
        ))}
      </div>
    </>
  );
}
