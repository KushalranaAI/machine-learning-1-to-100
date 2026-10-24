"use client";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import type { Nav } from "@/lib/nav";
import { AIProvider, useAI } from "./ai/AIProvider";
import AIPanel from "./ai/AIPanel";
import SelectionAsk from "./ai/SelectionAsk";
import CommandPalette from "./CommandPalette";
import Sidebar from "./Sidebar";
import Toaster from "./Toaster";
import Topbar from "./Topbar";

function Frame({ nav, children }: { nav: Nav; children: React.ReactNode }) {
  const pathname = usePathname();
  const ai = useAI();
  const [menu, setMenu] = useState(false);

  useEffect(() => setMenu(false), [pathname]);
  useEffect(() => {
    if (!menu) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setMenu(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [menu]);

  return (
    <div className="app" data-ai={ai.open ? "open" : "closed"}>
      <a href="#main" className="skip-link">Skip to content</a>
      <Sidebar nav={nav} open={menu} onNavigate={() => setMenu(false)} />
      {menu && <div className="scrim" onClick={() => setMenu(false)} aria-hidden="true" />}
      <div className="content">
        <Topbar nav={nav} onMenu={() => setMenu((m) => !m)} />
        <main id="main" className="main" key={pathname}>
          {children}
        </main>
      </div>
      {ai.open && <div className="scrim ai-scrim" onClick={ai.closePanel} aria-hidden="true" />}
      <AIPanel />
      <CommandPalette nav={nav} />
      <SelectionAsk />
      <Toaster />
    </div>
  );
}

export default function AppShell({ nav, children }: { nav: Nav; children: React.ReactNode }) {
  return (
    <AIProvider>
      <Frame nav={nav}>{children}</Frame>
    </AIProvider>
  );
}
