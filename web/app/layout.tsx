import "./globals.css";
import "katex/dist/katex.min.css";
import "highlight.js/styles/github-dark-dimmed.css";
import type { Metadata, Viewport } from "next";
import AppShell from "@/components/AppShell";
import { getNav } from "@/lib/navdata";
import { THEME_SCRIPT } from "@/lib/theme-script";

// Lessons are read from disk on every request, so edits show up without rebuilding.
export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: { default: "Machine Learning 1 → 100", template: "%s · ML 1 → 100" },
  description: "Learn classical machine learning from the math up - interactively, on your own machine.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f7f7fb" },
    { media: "(prefers-color-scheme: dark)", color: "#0c0e16" },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const nav = getNav();
  return (
    // suppressHydrationWarning: the inline script sets data-theme before React hydrates
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body>
        <AppShell nav={nav}>{children}</AppShell>
      </body>
    </html>
  );
}
