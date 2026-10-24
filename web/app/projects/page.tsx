import type { Metadata } from "next";
import ProjectsList from "@/components/ProjectsList";
import Icon from "@/components/ui/Icon";
import { getProjects } from "@/lib/content";

export const metadata: Metadata = { title: "Projects" };

export default function ProjectsPage() {
  const projects = getProjects().map((p) => ({ slug: p.slug, title: p.title, summary: p.summary, level: p.level, steps: p.steps.length, dataset: p.dataset.name, algorithms: p.algorithms }));
  return (
    <article>
      <header className="page-head">
        <div className="eyebrow"><Icon name="rocket" size={14} /> Build</div>
        <h1>Projects</h1>
        <p className="lead">Five end-to-end projects on real datasets, from a first regression to a neural network built from scratch. Tick off steps as you go.</p>
      </header>
      <ProjectsList projects={projects} />
    </article>
  );
}
