import type { Metadata } from "next";
import { notFound } from "next/navigation";
import AskButton from "@/components/ai/AskButton";
import Markdown from "@/components/Markdown";
import SectionLessons from "@/components/SectionLessons";
import SectionProgress from "@/components/progress/SectionProgress";
import Icon, { sectionIcon } from "@/components/ui/Icon";
import { getSectionBody, getSections } from "@/lib/content";

// Lets the GitHub Pages export pre-render every part; the local app still renders on request.
export function generateStaticParams() {
  return getSections().map((s) => ({ section: s.slug }));
}

export async function generateMetadata({ params }: { params: Promise<{ section: string }> }): Promise<Metadata> {
  const { section } = await params;
  return { title: getSections().find((x) => x.slug === section)?.title ?? "Part" };
}

export default async function SectionPage({ params }: { params: Promise<{ section: string }> }) {
  const { section } = await params;
  const s = getSections().find((x) => x.slug === section);
  if (!s) notFound();
  const body = getSectionBody(section);
  const nav = { slug: s.slug, title: s.title, order: s.order, description: s.description, lessons: s.lessons.map((l) => ({ slug: l.slug, title: l.title, summary: l.summary, minutes: l.minutes, level: l.level })) };
  return (
    <article>
      <header className="page-head with-ring">
        <div>
          <div className="eyebrow"><Icon name={sectionIcon(s.slug)} size={14} /> Part {s.order}</div>
          <h1>{s.title}</h1>
          <p className="lead">{s.description}</p>
          <div className="meta-row">
            <span className="pill">{s.lessons.length} lessons</span>
            <span className="pill"><Icon name="clock" size={13} /> {s.lessons.reduce((n, l) => n + l.minutes, 0)} min</span>
            <AskButton variant="ghost" label="What will I learn here?" prompt={`I'm about to start the part "${s.title}" of this course. In a few sentences, what will I be able to do after it, and how should I approach it?`} />
          </div>
        </div>
        <SectionProgress section={nav} />
      </header>

      <h2 className="section-title">Lessons</h2>
      <SectionLessons section={nav} />

      {body.trim() && (
        <>
          <h2 className="section-title">About this part</h2>
          <Markdown>{body}</Markdown>
        </>
      )}
    </article>
  );
}
