import Link from "next/link";
import { notFound } from "next/navigation";
import LessonView from "@/components/LessonView";
import Markdown from "@/components/Markdown";
import { flatLessons, getCode, getLesson, getProjects, getSections } from "@/lib/content";

export default async function LessonPage({ params }: { params: Promise<{ section: string; topic: string }> }) {
  const { section, topic } = await params;
  const lesson = getLesson(section, topic);
  if (!lesson) notFound();
  const sectionInfo = getSections().find((s) => s.slug === section)!;
  const all = flatLessons();
  const i = all.findIndex((l) => l.section === section && l.slug === topic);
  const prev = all[i - 1];
  const next = all[i + 1];
  const projects = getProjects();
  const code = lesson.code.map(getCode).filter((c) => c !== null);
  const demo = lesson.demo ? `${lesson.dir}/${lesson.demo}` : null;
  const lessonTitle = (id: string) => {
    const [s, t] = id.split("/");
    return all.find((l) => l.section === s && l.slug === t)?.title ?? id;
  };
  const usedIn = (ref: string) => {
    const p = projects.find((x) => ref.startsWith(x.slug));
    return p ? <Link href={`/projects/${p.slug}`}>{p.title}</Link> : <Markdown>{ref}</Markdown>;
  };

  return (
    <>
      <div className="stat">
        <Link href={`/learn/${section}`}>{sectionInfo.icon} {sectionInfo.title}</Link> · {lesson.minutes} min ·{" "}
        <span className={`pill ${lesson.level}`}>{lesson.level}</span>
      </div>
      <h1>{lesson.title}</h1>
      <p className="lead">{lesson.summary}</p>

      <div className="info-grid">
        {lesson.why && (
          <div className="card"><h4>Why you need it</h4><Markdown>{lesson.why}</Markdown></div>
        )}
        {lesson.background && (
          <div className="card"><h4>Background</h4><Markdown>{lesson.background}</Markdown></div>
        )}
        {lesson.use_cases.length > 0 && (
          <div className="card"><h4>Real-world use cases</h4><ul>{lesson.use_cases.map((u) => <li key={u}>{u}</li>)}</ul></div>
        )}
        {lesson.where_used.length > 0 && (
          <div className="card"><h4>Where it&apos;s used in this course</h4><ul>{lesson.where_used.map((u) => <li key={u}>{usedIn(u)}</li>)}</ul></div>
        )}
        {(lesson.pros.length > 0 || lesson.cons.length > 0) && (
          <div className="card">
            <h4>Strengths &amp; weaknesses</h4>
            <ul>
              {lesson.pros.map((p) => <li key={p}>✅ {p}</li>)}
              {lesson.cons.map((c) => <li key={c}>⚠️ {c}</li>)}
            </ul>
          </div>
        )}
        {lesson.prerequisites.length > 0 && (
          <div className="card">
            <h4>Before this lesson</h4>
            <ul>{lesson.prerequisites.map((p) => <li key={p}><Link href={`/learn/${p}`}>{lessonTitle(p)}</Link></li>)}</ul>
          </div>
        )}
      </div>

      <LessonView id={`${section}/${topic}`} body={lesson.body} workflow={lesson.workflow} widgets={lesson.widgets} code={code} demo={demo} />

      <div className="prevnext">
        {prev ? (
          <Link href={`/learn/${prev.section}/${prev.slug}`} className="card" style={{ color: "inherit", textDecoration: "none" }}>
            <div className="stat">← Previous</div>{prev.title}
          </Link>
        ) : <span />}
        {next ? (
          <Link href={`/learn/${next.section}/${next.slug}`} className="card" style={{ color: "inherit", textDecoration: "none", textAlign: "right" }}>
            <div className="stat">Next →</div>{next.title}
          </Link>
        ) : <span />}
      </div>
    </>
  );
}
