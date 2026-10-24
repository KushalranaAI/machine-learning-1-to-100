"use client";
import Link from "next/link";
import type { NavSection } from "@/lib/nav";
import { lessonId } from "@/lib/nav";
import { useProgress } from "@/lib/progress";
import Icon from "./ui/Icon";

/** The lessons of one part as a progress-aware list. */
export default function SectionLessons({ section }: { section: NavSection }) {
  const p = useProgress();
  return (
    <ol className="lesson-list">
      {section.lessons.map((l, i) => {
        const id = lessonId(section.slug, l.slug);
        const state = p.done[id] ? "done" : p.visited[id] ? "started" : "new";
        return (
          <li key={l.slug}>
            <Link href={`/learn/${section.slug}/${l.slug}`} className="lesson-row">
              <span className={`lesson-num ${state}`}>
                {state === "done" ? <Icon name="check" size={16} label="Completed" /> : section.order + "." + (i + 1)}
              </span>
              <span className="lesson-main">
                <b>{l.title}</b>
                <span className="muted small">{l.summary}</span>
              </span>
              <span className="lesson-meta">
                <span className={`pill ${l.level}`}>{l.level}</span>
                <span className="muted small"><Icon name="clock" size={12} /> {l.minutes} min</span>
              </span>
              <Icon name="chevron-right" size={16} className="go" />
            </Link>
          </li>
        );
      })}
    </ol>
  );
}
