// Server-only: builds the navigation/progress metadata the client components work from.
import { getProjects, getSections } from "./content";
import type { Nav } from "./nav";

export function getNav(): Nav {
  return {
    sections: getSections().map((s) => ({
      slug: s.slug,
      title: s.title,
      order: s.order,
      description: s.description,
      lessons: s.lessons.map((l) => ({ slug: l.slug, title: l.title, summary: l.summary, minutes: l.minutes, level: l.level })),
    })),
    projects: getProjects().map((p) => ({ slug: p.slug, title: p.title, summary: p.summary, level: p.level, steps: p.steps.length, dataset: p.dataset.name })),
  };
}
