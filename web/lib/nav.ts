// The slice of course metadata the client needs for navigation, search and progress. Built on the server (see layout.tsx).
export type NavLesson = { slug: string; title: string; summary: string; minutes: number; level: string };
export type NavSection = { slug: string; title: string; order: number; description: string; lessons: NavLesson[] };
export type NavProject = { slug: string; title: string; summary: string; level: string; steps: number; dataset: string };
export type Nav = { sections: NavSection[]; projects: NavProject[] };

export const lessonId = (section: string, lesson: string) => `${section}/${lesson}`;
export const projectId = (slug: string) => `project/${slug}`;
export const stepId = (slug: string, i: number) => `project/${slug}/${i}`;

export function allLessons(nav: Nav) {
  return nav.sections.flatMap((s) => s.lessons.map((l) => ({ ...l, section: s.slug, sectionTitle: s.title, id: lessonId(s.slug, l.slug), href: `/learn/${s.slug}/${l.slug}` })));
}
