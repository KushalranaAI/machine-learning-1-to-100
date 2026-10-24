"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { useProgress } from "./progress";

type Props = {
  sections: { slug: string; title: string; icon: string; lessons: { slug: string; title: string }[] }[];
  projects: { slug: string; title: string }[];
};

export default function Sidebar({ sections, projects }: Props) {
  const pathname = usePathname();
  const progress = useProgress();
  const [open, setOpen] = useState(false);
  return (
    <>
      <button className="btn menu-btn" onClick={() => setOpen(!open)} aria-label="Toggle navigation">☰</button>
      <nav className={`sidebar ${open ? "open" : ""}`} onClick={(e) => (e.target as HTMLElement).tagName === "A" && setOpen(false)}>
        <Link href="/" className="brand">
          ML 1 → 100<small>Machine learning from the math up</small>
        </Link>
        <div className="topnav">
          <Link href="/">Roadmap</Link>
          <Link href="/lab">Interactive lab</Link>
          <Link href="/projects">Projects</Link>
          <Link href="/map">Algorithm map</Link>
        </div>
        {sections.map((s) => {
          const inSection = pathname.startsWith(`/learn/${s.slug}`);
          const done = s.lessons.filter((l) => progress[`${s.slug}/${l.slug}`]).length;
          return (
            <details key={s.slug} className="nav-section" open={inSection}>
              <summary>
                <span>{s.icon}</span>
                <span style={{ flex: 1 }}>{s.title}</span>
                <span className="muted" style={{ fontSize: "0.75rem", fontWeight: 500 }}>{done}/{s.lessons.length}</span>
              </summary>
              <ul>
                {s.lessons.map((l) => {
                  const href = `/learn/${s.slug}/${l.slug}`;
                  return (
                    <li key={l.slug}>
                      <Link href={href} className={pathname === href ? "active" : ""}>
                        <span className={`done-dot ${progress[`${s.slug}/${l.slug}`] ? "done" : ""}`} />
                        {l.title}
                      </Link>
                    </li>
                  );
                })}
              </ul>
            </details>
          );
        })}
        <details className="nav-section" open={pathname.startsWith("/projects")}>
          <summary><span>🚀</span><span style={{ flex: 1 }}>Projects</span></summary>
          <ul>
            {projects.map((p) => (
              <li key={p.slug}>
                <Link href={`/projects/${p.slug}`} className={pathname === `/projects/${p.slug}` ? "active" : ""}>
                  <span className="done-dot" />{p.title}
                </Link>
              </li>
            ))}
          </ul>
        </details>
      </nav>
    </>
  );
}
