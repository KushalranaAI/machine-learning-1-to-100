"use client";
// A project's step-by-step guide with a checkbox per step, a run button for steps that have a script, and a tutor that knows the project.
import { useMemo } from "react";
import type { PageContext } from "@/lib/ai/prompts";
import { projectId, stepId } from "@/lib/nav";
import { setDone, useProgress } from "@/lib/progress";
import { toast } from "@/lib/toast";
import { usePageContext } from "./ai/AIProvider";
import AskButton from "./ai/AskButton";
import Markdown from "./Markdown";
import RunPanel from "./RunPanel";
import VisitTracker from "./progress/VisitTracker";
import Icon from "./ui/Icon";

type Step = { title: string; script?: string; description: string };

export default function ProjectSteps({ slug, title, dir, steps, problem, body }: { slug: string; title: string; dir: string; steps: Step[]; problem: string; body: string }) {
  const p = useProgress();
  const doneCount = steps.filter((_, i) => p.done[stepId(slug, i)]).length;

  const context = useMemo<PageContext>(
    () => ({
      key: `project:${slug}`,
      kind: "project",
      title,
      noteId: projectId(slug),
      text: `Project: ${title}\n\nProblem:\n${problem}\n\nSteps:\n${steps.map((s, i) => `${i + 1}. ${s.title}\n${s.description}`).join("\n\n")}\n\nWrite-up:\n${body}`,
    }),
    [slug, title, problem, steps, body],
  );
  usePageContext(context);

  return (
    <>
      <VisitTracker id={projectId(slug)} href={`/projects/${slug}`} title={title} />
      <div className="section-head">
        <h2>Step-by-step guide</h2>
        <span className="muted small">{doneCount} of {steps.length} steps done</span>
      </div>
      <div className="bar" style={{ marginBottom: 16 }} aria-hidden="true"><div style={{ width: `${steps.length ? (100 * doneCount) / steps.length : 0}%` }} /></div>
      <ol className="steps">
        {steps.map((step, i) => {
          const done = !!p.done[stepId(slug, i)];
          return (
            <li key={step.title} className={done ? "done" : ""}>
              <div className="step-head">
                <button
                  type="button"
                  className={`check ${done ? "on" : ""}`}
                  role="checkbox"
                  aria-checked={done}
                  aria-label={`Mark step ${i + 1} done: ${step.title}`}
                  onClick={() => {
                    setDone(stepId(slug, i), !done);
                    if (!done && doneCount + 1 === steps.length) toast("All steps done - mark the project complete when you're ready.", { tone: "info" });
                  }}
                >
                  {done ? <Icon name="check" size={16} /> : <span>{i + 1}</span>}
                </button>
                <h3>{step.title}</h3>
                <AskButton label="Help" prompt={`I'm on step ${i + 1} of this project: "${step.title}". Walk me through what to do, why, and what could go wrong.`} />
              </div>
              <Markdown>{step.description}</Markdown>
              {step.script && <RunPanel script={`${dir}/${step.script.split(" ")[0]}`} args={step.script.split(" ").slice(1)} label="Run this step" />}
            </li>
          );
        })}
      </ol>
    </>
  );
}
