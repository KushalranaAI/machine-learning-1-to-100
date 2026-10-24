"use client";
// Per-browser learning progress (which lessons are marked done). Lives in localStorage only.
import { useEffect, useState } from "react";

const KEY = "ml100-progress";
const EVENT = "ml100-progress-change";

function read(): Record<string, boolean> {
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? "{}");
  } catch {
    return {};
  }
}

export function setDone(id: string, done: boolean) {
  try {
    const p = read();
    if (done) p[id] = true;
    else delete p[id];
    localStorage.setItem(KEY, JSON.stringify(p));
    window.dispatchEvent(new Event(EVENT));
  } catch {
    /* storage unavailable: progress just isn't remembered */
  }
}

export function useProgress(): Record<string, boolean> {
  const [p, setP] = useState<Record<string, boolean>>({});
  useEffect(() => {
    const update = () => setP(read());
    update();
    window.addEventListener(EVENT, update);
    window.addEventListener("storage", update);
    return () => {
      window.removeEventListener(EVENT, update);
      window.removeEventListener("storage", update);
    };
  }, []);
  return p;
}
