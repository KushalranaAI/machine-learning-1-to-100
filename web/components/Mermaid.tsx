"use client";
import { useEffect, useId, useState } from "react";

let initialised = false;

export default function Mermaid({ chart }: { chart: string }) {
  const id = useId().replace(/[^a-zA-Z0-9]/g, "");
  const [svg, setSvg] = useState("");
  const [error, setError] = useState("");
  useEffect(() => {
    let cancelled = false;
    import("mermaid").then(async ({ default: mermaid }) => {
      if (!initialised) {
        const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
        mermaid.initialize({ startOnLoad: false, theme: dark ? "dark" : "default", securityLevel: "strict" });
        initialised = true;
      }
      try {
        const { svg } = await mermaid.render(`m${id}`, chart);
        if (!cancelled) setSvg(svg);
      } catch (e) {
        if (!cancelled) setError(String(e));
      }
    });
    return () => {
      cancelled = true;
    };
  }, [chart, id]);
  if (error) return <pre>{chart}</pre>;
  return <div className="mermaid-box" dangerouslySetInnerHTML={{ __html: svg || "<span class='muted'>drawing diagram…</span>" }} />;
}
