"use client";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import Mermaid from "./Mermaid";

/** Course markdown: GitHub tables, $math$ / $$math$$ (KaTeX), ```mermaid diagrams and highlighted code. */
export default function Markdown({ children }: { children: string }) {
  return (
    <div className="markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, [rehypeHighlight, { detect: false, ignoreMissing: true }]]}
        components={{
          code(props) {
            const { className, children, ...rest } = props;
            if (className?.includes("language-mermaid")) return <Mermaid chart={String(children).trim()} />;
            return <code className={className} {...rest}>{children}</code>;
          },
          pre(props) {
            const child = props.children as { props?: { className?: string } } | undefined;
            if (child?.props?.className?.includes("language-mermaid")) return <>{props.children}</>;
            return <pre>{props.children}</pre>;
          },
          a(props) {
            const external = props.href?.startsWith("http");
            return <a {...props} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined} />;
          },
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
