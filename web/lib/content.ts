// Reads the course content (curriculum/, projects/, source code) straight from the repository on disk.
// Markdown files are the single source of truth: GitHub renders them, and so does this app.
import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";

export const ROOT = path.resolve(process.cwd(), "..");
const CURRICULUM = path.join(ROOT, "curriculum");
const PROJECTS = path.join(ROOT, "projects");

export type Link = { label: string; href: string };

export type LessonMeta = {
  section: string;
  slug: string;
  title: string;
  order: number;
  summary: string;
  level: string;
  minutes: number;
  why: string;
  background: string;
  use_cases: string[];
  where_used: string[];
  prerequisites: string[];
  workflow: string;
  code: string[];
  demo: string | null;
  widgets: string[];
  pros: string[];
  cons: string[];
};

export type Lesson = LessonMeta & { body: string; dir: string };

export type Section = {
  slug: string;
  title: string;
  order: number;
  description: string;
  icon: string;
  lessons: LessonMeta[];
};

const asList = (v: unknown): string[] => (Array.isArray(v) ? v.map(String) : v ? [String(v)] : []);

function readDirs(dir: string): string[] {
  if (!fs.existsSync(dir)) return [];
  return fs
    .readdirSync(dir, { withFileTypes: true })
    .filter((d) => d.isDirectory() && /^\d\d-/.test(d.name))
    .map((d) => d.name)
    .sort();
}

function lessonMeta(section: string, slug: string, data: Record<string, unknown>): LessonMeta {
  return {
    section,
    slug,
    title: String(data.title ?? slug),
    order: Number(data.order ?? parseInt(slug, 10)),
    summary: String(data.summary ?? ""),
    level: String(data.level ?? "beginner"),
    minutes: Number(data.minutes ?? 30),
    why: String(data.why ?? ""),
    background: String(data.background ?? ""),
    use_cases: asList(data.use_cases),
    where_used: asList(data.where_used),
    prerequisites: asList(data.prerequisites),
    workflow: String(data.workflow ?? ""),
    code: asList(data.code),
    demo: data.demo ? String(data.demo) : null,
    widgets: asList(data.widgets ?? data.widget),
    pros: asList(data.pros),
    cons: asList(data.cons),
  };
}

export function getSections(): Section[] {
  return readDirs(CURRICULUM).map((section) => {
    const readme = path.join(CURRICULUM, section, "README.md");
    const { data } = fs.existsSync(readme) ? matter(fs.readFileSync(readme, "utf8")) : { data: {} as Record<string, unknown> };
    const lessons = readDirs(path.join(CURRICULUM, section))
      .map((slug) => {
        const file = path.join(CURRICULUM, section, slug, "README.md");
        if (!fs.existsSync(file)) return null;
        return lessonMeta(section, slug, matter(fs.readFileSync(file, "utf8")).data);
      })
      .filter((l): l is LessonMeta => l !== null)
      .sort((a, b) => a.order - b.order);
    return {
      slug: section,
      title: String(data.title ?? section),
      order: parseInt(section, 10),
      description: String(data.description ?? ""),
      icon: String(data.icon ?? "📘"),
      lessons,
    };
  });
}

export function getSectionBody(section: string): string {
  const file = path.join(CURRICULUM, section, "README.md");
  return fs.existsSync(file) ? matter(fs.readFileSync(file, "utf8")).content : "";
}

export function getLesson(section: string, slug: string): Lesson | null {
  const dir = path.join(CURRICULUM, section, slug);
  const file = path.join(dir, "README.md");
  if (!safeInside(CURRICULUM, file) || !fs.existsSync(file)) return null;
  const { data, content } = matter(fs.readFileSync(file, "utf8"));
  return { ...lessonMeta(section, slug, data), body: content, dir: path.relative(ROOT, dir) };
}

export function flatLessons(): LessonMeta[] {
  return getSections().flatMap((s) => s.lessons);
}

// ------------------------------------------------------------------------------------- projects

export type ProjectStep = { title: string; script?: string; description: string };
export type Project = {
  slug: string;
  title: string;
  order: number;
  level: string;
  summary: string;
  problem: string;
  dataset: { name: string; source: string; url: string; kaggle?: string; license?: string; size?: string };
  algorithms: string[];
  skills: string[];
  lessons: string[];
  steps: ProjectStep[];
  workflow: string;
  body: string;
  dir: string;
};

export function getProjects(): Project[] {
  return readDirs(PROJECTS)
    .map((slug) => getProject(slug))
    .filter((p): p is Project => p !== null);
}

export function getProject(slug: string): Project | null {
  const dir = path.join(PROJECTS, slug);
  const file = path.join(dir, "README.md");
  if (!safeInside(PROJECTS, file) || !fs.existsSync(file)) return null;
  const { data, content } = matter(fs.readFileSync(file, "utf8"));
  return {
    slug,
    title: String(data.title ?? slug),
    order: parseInt(slug, 10),
    level: String(data.level ?? ""),
    summary: String(data.summary ?? ""),
    problem: String(data.problem ?? ""),
    dataset: (data.dataset as Project["dataset"]) ?? { name: "", source: "", url: "" },
    algorithms: asList(data.algorithms),
    skills: asList(data.skills),
    lessons: asList(data.lessons),
    steps: (data.steps as ProjectStep[]) ?? [],
    workflow: String(data.workflow ?? ""),
    body: content,
    dir: path.relative(ROOT, dir),
  };
}

// ---------------------------------------------------------------------------------- source code

export type CodeSnippet = { spec: string; file: string; symbol: string | null; language: string; code: string };

const LANG: Record<string, string> = { ".py": "python", ".cpp": "cpp", ".hpp": "cpp", ".h": "cpp", ".txt": "cmake" };

export function safeInside(base: string, target: string): boolean {
  const rel = path.relative(base, path.resolve(target));
  return !rel.startsWith("..") && !path.isAbsolute(rel);
}

/** "mlscratch/linear_model.py:Ridge" → the source of class Ridge; "cpp/src/knn.cpp" → the whole file. */
export function getCode(spec: string): CodeSnippet | null {
  const [file, symbol = null] = spec.split(":");
  const full = path.join(ROOT, file);
  if (!safeInside(ROOT, full) || !fs.existsSync(full)) return null;
  const source = fs.readFileSync(full, "utf8");
  const language = LANG[path.extname(file)] ?? "text";
  return { spec, file, symbol, language, code: symbol && language === "python" ? extractPython(source, symbol) : source };
}

/** Extract a top-level class/function (or Class.method) including decorators, by indentation. */
export function extractPython(source: string, symbol: string): string {
  const lines = source.split("\n");
  let indent = "";
  let start = -1;
  const parts = symbol.split(".");
  let searchFrom = 0;
  for (const [depth, name] of parts.entries()) {
    const pad = " ".repeat(depth * 4);
    const re = new RegExp(`^${pad}(class|def|async def)\\s+${name}\\b`);
    start = lines.findIndex((l, i) => i >= searchFrom && re.test(l));
    if (start < 0) return `# ${symbol} not found in this file`;
    searchFrom = start + 1;
    indent = pad;
  }
  while (start > 0 && lines[start - 1].startsWith(`${indent}@`)) start--;
  let end = start + 1;
  while (end < lines.length) {
    const l = lines[end];
    if (l.trim() !== "" && !l.startsWith(indent + " ") && !l.startsWith(indent + "\t") && !l.startsWith(indent + ")")) break;
    end++;
  }
  while (end > start && lines[end - 1].trim() === "") end--;
  return lines.slice(start, end).join("\n");
}
