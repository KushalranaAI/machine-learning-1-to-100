"use client";
// A lesson as a guided journey: Overview → Learn → Workflow → Interact → Code → Run → Practice → Notes.
// One step on screen at a time (progressive disclosure), with a clear "next" at the end of each.
import { useEffect, useMemo, useRef, useState } from "react";
import { type PageContext } from "@/lib/ai/prompts";
import Link from "next/link";
import { markVisited, progressStore, useProgress } from "@/lib/progress";
import { useResolvedTheme } from "@/lib/theme";
import { usePageContext } from "./ai/AIProvider";
import AskButton from "./ai/AskButton";
import Notes from "./lesson/Notes";
import Quiz from "./lesson/Quiz";
import Toc from "./lesson/Toc";
import Markdown from "./Markdown";
import Mermaid from "./Mermaid";
import MarkComplete from "./progress/MarkComplete";
import RunPanel from "./RunPanel";
import Icon, { type IconName } from "./ui/Icon";
import { WIDGETS } from "./widgets";

type Snippet = { spec: string; file: string; symbol: string | null; language: string; code: string };

type Props = {
  id: string;
  href: string;
  title: string;
  sectionTitle: string;
  summary: string;
  overview: React.ReactNode;
  body: string;
  workflow: string;
  widgets: string[];
  code: Snippet[];
  demo: string | null;
  next: { href: string; title: string } | null;
};

const prettify = (id: string) => id.replace(/-/g, " ").replace(/^./, (c) => c.toUpperCase());

export default function LessonView({ id, href, title, sectionTitle, summary, overview, body, workflow, widgets, code, demo, next }: Props) {
  const steps = useMemo(
    () =>
      [
        { key: "overview", label: "Overview", icon: "lightbulb" as IconName },
        { key: "learn", label: "Learn", icon: "book" as IconName },
        ...(workflow ? [{ key: "workflow", label: "Workflow", icon: "workflow" as IconName }] : []),
        ...(widgets.length ? [{ key: "interact", label: "Interact", icon: "sliders" as IconName }] : []),
        ...(code.length ? [{ key: "code", label: "Code", icon: "code" as IconName }] : []),
        ...(demo ? [{ key: "run", label: "Run", icon: "play" as IconName }] : []),
        { key: "practice", label: "Practice", icon: "target" as IconName },
        { key: "notes", label: "Notes", icon: "pen" as IconName },
      ],
    [workflow, widgets.length, code.length, demo],
  );
  const progress = useProgress();
  const theme = useResolvedTheme();
  const [step, setStep] = useState<string | null>(null);
  const [seen, setSeen] = useState<Set<string>>(new Set());
  const initial = useRef<{ id: string; step: string } | null>(null);

  // Start where it makes sense: the URL hash, else Learn for returning learners, else the Overview.
  useEffect(() => {
    const fromHash = window.location.hash.slice(1);
    // Decide once per lesson (effects can run twice in development), before recording this visit.
    if (initial.current?.id !== id) {
      const returning = !!progressStore.get().visited[id];
      initial.current = { id, step: steps.some((s) => s.key === fromHash) ? fromHash : returning ? "learn" : "overview" };
      markVisited(id, href, title);
    }
    setStep(initial.current.step);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  useEffect(() => {
    if (!step) return;
    document.getElementById(`tab-${step}`)?.scrollIntoView({ block: "nearest", inline: "nearest" });
    setSeen((s) => new Set(s).add(step));
    if (window.location.hash.slice(1) !== step) history.replaceState(null, "", `#${step}`);
  }, [step]);

  const go = (key: string) => {
    setStep(key);
    document.getElementById("lesson-top")?.scrollIntoView({ block: "start", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  };

  // what the AI tutor can see
  const codeText = useMemo(() => code.map((c) => `// ${c.file}${c.symbol ? ` → ${c.symbol}` : ""}\n${c.code}`).join("\n\n"), [code]);
  const current = steps.find((s) => s.key === step);
  const context = useMemo<PageContext>(() => {
    const head = `Lesson: ${title}\nPart: ${sectionTitle}\nSummary: ${summary}`;
    const onCode = step === "code";
    return {
      key: `lesson:${id}`,
      kind: "lesson",
      title,
      noteId: id,
      focus: current ? `the “${current.label}” step${onCode ? " (the from-scratch code)" : step === "interact" ? ` (interactive visuals: ${widgets.map(prettify).join(", ")})` : step === "workflow" ? " (the ML workflow diagram)" : ""}` : undefined,
      text: onCode
        ? `${head}\n\n# From-scratch code on screen\n${codeText}\n\n# Lesson text\n${body}`
        : `${head}\n\n# Lesson text\n${body}${step === "workflow" && workflow ? `\n\n# Workflow diagram (mermaid)\n${workflow}` : ""}${codeText ? `\n\n# Code in this lesson\n${codeText}` : ""}`,
    };
  }, [id, title, sectionTitle, summary, body, workflow, widgets, codeText, step, current]);
  usePageContext(step ? context : null);

  const idx = steps.findIndex((s) => s.key === step);
  const prev = idx > 0 ? steps[idx - 1] : null;
  const following = idx >= 0 && idx < steps.length - 1 ? steps[idx + 1] : null;
  const done = !!progress.done[id];

  if (!step) {
    return (
      <div aria-busy="true">
        <div className="skeleton" style={{ height: 44, marginBottom: 20 }} />
        <div className="skeleton" style={{ height: 260 }} />
      </div>
    );
  }

  return (
    <div id="lesson-top" className="lesson">
      <div className="stepper" role="tablist" aria-label="Lesson steps" onKeyDown={(e) => {
        if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
        e.preventDefault();
        const n = steps[(idx + (e.key === "ArrowRight" ? 1 : -1) + steps.length) % steps.length];
        setStep(n.key);
        requestAnimationFrame(() => document.getElementById(`tab-${n.key}`)?.focus());
      }}>
        {steps.map((s, i) => (
          <button
            key={s.key}
            id={`tab-${s.key}`}
            role="tab"
            type="button"
            aria-selected={step === s.key}
            aria-controls="lesson-panel"
            tabIndex={step === s.key ? 0 : -1}
            className={`step ${step === s.key ? "active" : ""} ${seen.has(s.key) && step !== s.key ? "seen" : ""}`}
            onClick={() => go(s.key)}
          >
            <span className="step-num">{seen.has(s.key) && step !== s.key ? <Icon name="check" size={13} /> : i + 1}</span>
            <Icon name={s.icon} size={16} className="step-icon" />
            <span className="step-label">{s.label}</span>
          </button>
        ))}
      </div>

      <div id="lesson-panel" role="tabpanel" aria-labelledby={`tab-${step}`} className="step-panel" tabIndex={-1}>
        {step === "overview" && <div className="info-grid">{overview}</div>}

        {step === "learn" && (
          <div className="with-toc">
            <div id="lesson-learn" className="learn-col"><Markdown>{body}</Markdown></div>
            <Toc rootId="lesson-learn" />
          </div>
        )}

        {step === "workflow" && (
          <div>
            <p className="muted">Where this fits in a real machine-learning workflow:</p>
            <Mermaid chart={workflow} />
          </div>
        )}

        {step === "interact" && (
          <div>
            <p className="muted">Drag, slide and step through it. Each visual runs a real implementation in your browser.</p>
            {widgets.map((w) => {
              const Widget = WIDGETS[w];
              return (
                <section key={w} className="widget-frame">
                  <header>
                    <h3>{prettify(w)}</h3>
                    <AskButton
                      label="Ask about this"
                      prompt={`Explain what the interactive visual “${prettify(w)}” on this page demonstrates, and give me three things to try with its controls and what I should notice each time.`}
                    />
                  </header>
                  {Widget ? <Widget key={theme} /> : <p className="muted">This visual isn&apos;t available.</p>}
                </section>
              );
            })}
          </div>
        )}

        {step === "code" && (
          <div>
            <p className="muted">The exact code in this repository - no library hiding the math. Open it in your editor and change something.</p>
            {code.map((c) => (
              <section key={c.spec} className="code-file">
                <div className="code-file-head">
                  <Icon name="code" size={16} />
                  <code>{c.file}</code>
                  {c.symbol && <span className="pill accent">{c.symbol}</span>}
                </div>
                <Markdown>{"```" + c.language + "\n" + c.code + "\n```"}</Markdown>
              </section>
            ))}
          </div>
        )}

        {step === "run" && demo && (
          <div>
            <p className="muted">
              Runs <code>{demo.split("/").pop()}</code> with the Python you started this app with - same as a terminal. Output and figures appear below.
            </p>
            <RunPanel script={demo} />
          </div>
        )}

        {step === "practice" && <Quiz id={id} title={title} text={body} />}
        {step === "notes" && <Notes id={id} />}
      </div>

      <div className="step-nav">
        {prev ? (
          <button type="button" className="btn" onClick={() => go(prev.key)}>
            <Icon name="arrow-left" size={16} /> {prev.label}
          </button>
        ) : <span />}
        {following ? (
          <button type="button" className="btn primary" onClick={() => go(following.key)}>
            Next: {following.label} <Icon name="arrow-right" size={16} />
          </button>
        ) : (
          <div className="finish">
            <MarkComplete id={id} next={next} size="lg" />
          </div>
        )}
      </div>
      {!following && done && next && (
        <div className="up-next card">
          <span className="muted small">Up next</span>
          <Link href={next.href}><b>{next.title}</b> <Icon name="arrow-right" size={16} /></Link>
        </div>
      )}
    </div>
  );
}
