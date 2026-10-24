import "./globals.css";
import "katex/dist/katex.min.css";
import "highlight.js/styles/github-dark-dimmed.css";
import type { Metadata } from "next";
import Sidebar from "@/components/Sidebar";
import { getProjects, getSections } from "@/lib/content";

// Lessons are read from disk on every request, so edits show up without rebuilding.
export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Machine Learning 1 → 100",
  description: "Learn classical machine learning from the math up - interactively, on your own machine.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const sections = getSections().map((s) => ({
    slug: s.slug,
    title: s.title,
    icon: s.icon,
    lessons: s.lessons.map((l) => ({ slug: l.slug, title: l.title })),
  }));
  const projects = getProjects().map((p) => ({ slug: p.slug, title: p.title }));
  return (
    <html lang="en">
      <body>
        <div className="shell">
          <Sidebar sections={sections} projects={projects} />
          <main className="main">{children}</main>
        </div>
      </body>
    </html>
  );
}
