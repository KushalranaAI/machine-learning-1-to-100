// What the tutor is told about the learner's current page, and quick prompts offered per page type.

export type PageContext = {
  key: string; // chat thread id, e.g. "lesson:07-regression/01-linear-regression"
  kind: "lesson" | "project" | "map" | "lab" | "section" | "general";
  title: string;
  /** Progress id where "save to notes" can write, if any. */
  noteId?: string;
  /** What the learner is looking at right now (the lesson step, a widget …). */
  focus?: string;
  /** The page's own text: lesson body, code, project steps. Trimmed before sending. */
  text: string;
};

export const GENERAL: PageContext = { key: "general", kind: "general", title: "Machine learning", text: "" };

const BASE = `You are a patient, precise machine-learning tutor built into "Machine Learning 1 → 100", a hands-on course that implements classical ML from the math up in Python/NumPy (package "mlscratch") and C++.

How to answer:
- Teach, don't just tell. Start with the intuition in one or two sentences, then the math or code if it helps.
- Use Markdown. Write math with $...$ (inline) and $$...$$ (display) - it is rendered with KaTeX. Put code in fenced blocks with a language tag.
- Be concise: short paragraphs, small numeric examples, no filler. Offer a follow-up only when it is genuinely useful.
- When the learner's page text is provided below, treat it as the source of truth for this course's notation and code; refer to its sections and file names. If the answer is not covered there, say so and answer from general knowledge.
- If something the learner says is wrong, correct it kindly and explain why.
- If you are unsure, say so rather than guessing.`;

export function buildSystem(ctx: PageContext, share: boolean, budget: number): string {
  if (!share || ctx.kind === "general") return BASE;
  const head = `\n\nThe learner is on: ${ctx.title} (${ctx.kind})${ctx.focus ? `\nThey are currently looking at: ${ctx.focus}` : ""}`;
  const text = ctx.text.length > budget ? ctx.text.slice(0, budget) + "\n…[page text trimmed]" : ctx.text;
  return `${BASE}${head}\n\n<page>\n${text}\n</page>`;
}

export function buildUserTurn(text: string, quote?: string): string {
  if (!quote) return text;
  return `Here is the excerpt I'm asking about:\n\n"""\n${quote}\n"""\n\n${text}`;
}

export type QuickPrompt = { label: string; prompt: string };

export function quickPrompts(ctx: PageContext): QuickPrompt[] {
  if (ctx.kind === "lesson") {
    const code = ctx.focus?.toLowerCase().includes("code");
    return [
      ...(code
        ? [
            { label: "Explain this code line by line", prompt: "Explain the from-scratch code on this page line by line. Connect each part to the math." },
            { label: "Map code to the math", prompt: "Show how each part of the code corresponds to the equations in the lesson." },
          ]
        : [
            { label: "Explain it like I'm new", prompt: "Explain this lesson's main idea like I'm new to it. Use one concrete everyday analogy." },
            { label: "Walk through the math", prompt: "Walk me through the key derivation on this page step by step, explaining each step." },
          ]),
      { label: "Common mistakes", prompt: "What are the most common mistakes or misconceptions about this topic, and how do I avoid them?" },
      { label: "Real-world example", prompt: "Give me a small, concrete real-world example with actual numbers that I can work through by hand." },
      { label: "Quiz me", prompt: "Quiz me with 3 questions on this lesson, one at a time. Wait for my answer before revealing the solution." },
    ];
  }
  if (ctx.kind === "project") {
    return [
      { label: "Where do I start?", prompt: "I'm about to start this project. What should I think about first, and what are the likely pitfalls?" },
      { label: "Explain the approach", prompt: "Explain the overall approach of this project and why each step is there." },
      { label: "Which metric & why?", prompt: "Which evaluation metric fits this problem and why? What would a good result look like?" },
    ];
  }
  if (ctx.kind === "map") {
    return [
      { label: "Help me choose", prompt: "I'll describe my problem; help me choose an algorithm from this map. What do you need to know about my data?" },
      { label: "Compare top contenders", prompt: "Compare the algorithms currently listed: when would I pick each one, and what are the trade-offs?" },
    ];
  }
  if (ctx.kind === "lab") {
    return [
      { label: "What should I try?", prompt: "What are three experiments I can do with this visual to build intuition, and what should I notice each time?" },
      { label: "What am I seeing?", prompt: "Explain what this visual is demonstrating and how to read it." },
    ];
  }
  return [
    { label: "Where should I start?", prompt: "I'm new to machine learning. Suggest a learning path through this course for me, based on common goals." },
    { label: "Explain overfitting", prompt: "Explain overfitting in plain language with a simple example." },
    { label: "Supervised vs unsupervised", prompt: "What is the difference between supervised and unsupervised learning? Give examples." },
  ];
}
