"use client";
// Ctrl/Cmd+K: jump to any lesson, algorithm, project or page; run quick actions. Recognition beats recall.
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { ALGORITHMS } from "@/lib/algorithms";
import { type Nav, allLessons } from "@/lib/nav";
import { applyTheme, getThemePref } from "@/lib/theme";
import { useProgress } from "@/lib/progress";
import { useAI } from "./ai/AIProvider";
import Icon, { type IconName } from "./ui/Icon";

type Item = { id: string; group: string; title: string; sub?: string; icon: IconName; keywords?: string; run: () => void };

export const PALETTE_EVENT = "ml100-open-palette";

export default function CommandPalette({ nav }: { nav: Nav }) {
  const router = useRouter();
  const ai = useAI();
  const progress = useProgress();
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const [active, setActive] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const list = useRef<HTMLUListElement>(null);
  const returnFocus = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((o) => !o);
      }
    };
    const onOpen = () => setOpen(true);
    window.addEventListener("keydown", onKey);
    window.addEventListener(PALETTE_EVENT, onOpen);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener(PALETTE_EVENT, onOpen);
    };
  }, []);

  useEffect(() => {
    if (open) {
      returnFocus.current = document.activeElement as HTMLElement | null;
      setQ("");
      setActive(0);
      requestAnimationFrame(() => input.current?.focus());
    } else returnFocus.current?.focus?.();
  }, [open]);

  const items = useMemo<Item[]>(() => {
    const go = (href: string) => () => router.push(href);
    const pages: Item[] = [
      { id: "p-home", group: "Pages", title: "Dashboard", icon: "home", run: go("/") },
      { id: "p-progress", group: "Pages", title: "My progress", icon: "trophy", keywords: "streak stats history", run: go("/progress") },
      { id: "p-lab", group: "Pages", title: "Interactive lab", icon: "flask", keywords: "visualisations widgets playground", run: go("/lab") },
      { id: "p-map", group: "Pages", title: "Algorithm map", icon: "map", keywords: "which algorithm choose compare", run: go("/map") },
      { id: "p-projects", group: "Pages", title: "Projects", icon: "rocket", run: go("/projects") },
    ];
    const lessons: Item[] = allLessons(nav).map((l) => ({
      id: `l-${l.id}`,
      group: "Lessons",
      title: l.title,
      sub: l.sectionTitle,
      icon: "book",
      keywords: l.summary,
      run: go(l.href),
    }));
    const algos: Item[] = ALGORITHMS.map((a) => ({
      id: `a-${a.name}`,
      group: "Algorithms",
      title: a.name,
      sub: a.tasks.join(" · "),
      icon: "network",
      keywords: `${a.idea} ${a.strengths}`,
      run: go(`/learn/${a.lesson}`),
    }));
    const projects: Item[] = nav.projects.map((p) => ({
      id: `pr-${p.slug}`,
      group: "Projects",
      title: p.title,
      sub: p.dataset,
      icon: "rocket",
      keywords: p.summary,
      run: go(`/projects/${p.slug}`),
    }));
    const actions: Item[] = [
      { id: "x-ai", group: "Actions", title: "Open AI tutor", sub: "Ctrl + /", icon: "sparkles", keywords: "ask chat help", run: () => ai.openPanel("chat") },
      { id: "x-ai-set", group: "Actions", title: "AI settings (connect Ollama, Claude, OpenAI, Gemini)", icon: "key", keywords: "api key model local", run: () => ai.openPanel("settings") },
      {
        id: "x-theme",
        group: "Actions",
        title: "Cycle theme (system / light / dark)",
        icon: "moon",
        run: () => {
          const order = ["system", "light", "dark"] as const;
          applyTheme(order[(order.indexOf(getThemePref()) + 1) % 3]);
        },
      },
    ];
    return [...pages, ...actions, ...lessons, ...algos, ...projects];
  }, [nav, router, ai]);

  const results = useMemo(() => {
    const tokens = q.toLowerCase().split(/\s+/).filter(Boolean);
    if (!tokens.length) {
      const last = progress.last;
      const recent: Item[] = last ? [{ id: "recent", group: "Continue", title: last.title, sub: "Where you left off", icon: "play", run: () => router.push(last.href) }] : [];
      return [...recent, ...items.filter((i) => i.group === "Pages" || i.group === "Actions")];
    }
    return items
      .map((it) => {
        const title = it.title.toLowerCase();
        const hay = `${title} ${(it.sub ?? "").toLowerCase()} ${(it.keywords ?? "").toLowerCase()}`;
        if (!tokens.every((t) => hay.includes(t))) return null;
        const score = tokens.reduce((s, t) => s + (title.startsWith(t) ? 3 : title.includes(t) ? 2 : 0), 0);
        return { it, score };
      })
      .filter((x): x is { it: Item; score: number } => x !== null)
      .sort((a, b) => b.score - a.score)
      .slice(0, 40)
      .map((x) => x.it);
  }, [q, items, progress.last, router]);

  useEffect(() => setActive(0), [q]);
  useEffect(() => {
    list.current?.querySelector<HTMLElement>(`[data-i="${active}"]`)?.scrollIntoView({ block: "nearest" });
  }, [active]);

  if (!open) return null;
  const choose = (it: Item | undefined) => {
    if (!it) return;
    setOpen(false);
    it.run();
  };

  let lastGroup = "";
  return (
    <div className="palette-scrim" onMouseDown={(e) => e.target === e.currentTarget && setOpen(false)}>
      <div className="palette" role="dialog" aria-modal="true" aria-label="Search the course">
        <div className="palette-input">
          <Icon name="search" />
          <input
            ref={input}
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search lessons, algorithms, projects…"
            role="combobox"
            aria-expanded="true"
            aria-controls="palette-list"
            aria-activedescendant={results[active] ? `pal-${active}` : undefined}
            autoComplete="off"
            spellCheck={false}
            onKeyDown={(e) => {
              if (e.key === "ArrowDown") { e.preventDefault(); setActive((a) => Math.min(a + 1, results.length - 1)); }
              else if (e.key === "ArrowUp") { e.preventDefault(); setActive((a) => Math.max(a - 1, 0)); }
              else if (e.key === "Enter") { e.preventDefault(); choose(results[active]); }
              else if (e.key === "Escape") setOpen(false);
            }}
          />
          <kbd>Esc</kbd>
        </div>
        <ul id="palette-list" className="palette-list" role="listbox" ref={list}>
          {results.length === 0 && <li className="palette-empty">Nothing matches “{q}”. Try a shorter word, or ask the AI tutor.</li>}
          {results.map((it, i) => {
            const header = it.group !== lastGroup;
            lastGroup = it.group;
            return (
              <li key={it.id} role="presentation">
                {header && <div className="palette-group">{it.group}</div>}
                <div
                  id={`pal-${i}`}
                  data-i={i}
                  role="option"
                  aria-selected={i === active}
                  className={`palette-item ${i === active ? "active" : ""}`}
                  onMouseMove={() => setActive(i)}
                  onClick={() => choose(it)}
                >
                  <Icon name={it.icon} />
                  <span className="palette-title">{it.title}</span>
                  {it.sub && <span className="muted small">{it.sub}</span>}
                </div>
              </li>
            );
          })}
        </ul>
        <div className="palette-foot muted small">
          <span><kbd>↑</kbd><kbd>↓</kbd> move</span>
          <span><kbd>Enter</kbd> open</span>
          <span><kbd>Ctrl</kbd>+<kbd>/</kbd> AI tutor</span>
        </div>
      </div>
    </div>
  );
}
