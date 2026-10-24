import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import AskButton from "@/components/ai/AskButton";
import LessonView from "@/components/LessonView";
import Markdown from "@/components/Markdown";
import LessonStatus from "@/components/progress/LessonStatus";
import Icon, { type IconName, sectionIcon } from "@/components/ui/Icon";
import { flatLessons, getCode, getLesson, getProjects, getSections } from "@/lib/content";

export async function generateMetadata({ params }: { params: Promise<{ section: string; topic: string }> }): Promise<Metadata> {
  const { section, topic } = await params;
  return { title: getLesson(section, topic)?.title ?? "Lesson" };
}

function OverviewCard({ icon, title, wide, children }: { icon: IconName; title: string; wide?: boolean; children: React.ReactNode }) {
  return (
    <section className={`card overview-card ${wide ? "wide" : ""}`}>
      <h4><Icon name={icon} size={16} /> {title}</h4>
      {children}
    </section>
  );
}

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
  const demo = lesson.demo ? `${lesson.dir.replace(/\\/g, "/")}/${lesson.demo}` : null;
  const lessonTitle = (id: string) => {
    const [s, t] = id.split("/");
    return all.find((l) => l.section === s && l.slug === t)?.title ?? id;
  };
  const usedIn = (ref: string) => {
    const p = projects.find((x) => ref.startsWith(x.slug));
    return p ? <Link href={`/projects/${p.slug}`}>{p.title}</Link> : <Markdown>{ref}</Markdown>;
  };

  const overview = (
    <>
      {lesson.why && (
        <OverviewCard icon="lightbulb" title="Why you need it" wide>
          <Markdown>{lesson.why}</Markdown>
        </OverviewCard>
      )}
      {lesson.background && (
        <OverviewCard icon="book" title="Where it comes from">
          <Markdown>{lesson.background}</Markdown>
        </OverviewCard>
      )}
      {lesson.use_cases.length > 0 && (
        <OverviewCard icon="target" title="Real-world use cases">
          <ul>{lesson.use_cases.map((u) => <li key={u}>{u}</li>)}</ul>
        </OverviewCard>
      )}
      {(lesson.pros.length > 0 || lesson.cons.length > 0) && (
        <OverviewCard icon="layers" title="Strengths & weaknesses">
          <ul className="proscons">
            {lesson.pros.map((p) => <li key={p} className="pro"><Icon name="check" size={15} label="Strength" /> {p}</li>)}
            {lesson.cons.map((c) => <li key={c} className="con"><Icon name="alert" size={15} label="Weakness" /> {c}</li>)}
          </ul>
        </OverviewCard>
      )}
      {lesson.prerequisites.length > 0 && (
        <OverviewCard icon="route" title="Before this lesson">
          <ul>{lesson.prerequisites.map((p) => <li key={p}><Link href={`/learn/${p}`}>{lessonTitle(p)}</Link></li>)}</ul>
        </OverviewCard>
      )}
      {lesson.where_used.length > 0 && (
        <OverviewCard icon="rocket" title="Used in this course">
          <ul>{lesson.where_used.map((u) => <li key={u}>{usedIn(u)}</li>)}</ul>
        </OverviewCard>
      )}
    </>
  );

  const id = `${section}/${topic}`;
  return (
    <article>
      <header className="page-head">
        <div className="eyebrow">
          <Link href={`/learn/${section}`}><Icon name={sectionIcon(section)} size={14} /> Part {sectionInfo.order} · {sectionInfo.title}</Link>
        </div>
        <h1>{lesson.title}</h1>
        <p className="lead">{lesson.summary}</p>
        <div className="meta-row">
          <span className={`pill ${lesson.level}`}>{lesson.level}</span>
          <span className="pill"><Icon name="clock" size={13} /> {lesson.minutes} min</span>
          {lesson.widgets.length > 0 && <span className="pill accent"><Icon name="sliders" size={13} /> {lesson.widgets.length} interactive</span>}
          {lesson.demo && <span className="pill"><Icon name="play" size={13} /> runnable demo</span>}
          <LessonStatus id={id} />
          <span className="spacer" />
          <AskButton variant="ghost" label="Quick summary" prompt="Give me a 5-line summary of this lesson: the problem it solves, the core idea, the key equation or step, when to use it, and one pitfall." />
        </div>
      </header>

      <LessonView
        id={id}
        href={`/learn/${section}/${topic}`}
        title={lesson.title}
        sectionTitle={sectionInfo.title}
        summary={lesson.summary}
        overview={overview}
        body={lesson.body}
        workflow={lesson.workflow}
        widgets={lesson.widgets}
        code={code}
        demo={demo}
        next={next ? { href: `/learn/${next.section}/${next.slug}`, title: next.title } : null}
      />

      <nav className="prevnext" aria-label="Previous and next lesson">
        {prev ? (
          <Link href={`/learn/${prev.section}/${prev.slug}`} className="card link-card">
            <span className="muted small"><Icon name="arrow-left" size={14} /> Previous</span>
            <b>{prev.title}</b>
          </Link>
        ) : <span />}
        {next ? (
          <Link href={`/learn/${next.section}/${next.slug}`} className="card link-card right">
            <span className="muted small">Next <Icon name="arrow-right" size={14} /></span>
            <b>{next.title}</b>
          </Link>
        ) : <span />}
      </nav>
    </article>
  );
}
