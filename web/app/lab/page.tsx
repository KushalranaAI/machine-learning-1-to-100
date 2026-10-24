import type { Metadata } from "next";
import Lab from "@/components/Lab";
import Icon from "@/components/ui/Icon";

export const metadata: Metadata = { title: "Interactive lab" };

export default function LabPage() {
  return (
    <article>
      <header className="page-head">
        <div className="eyebrow"><Icon name="flask" size={14} /> Play</div>
        <h1>Interactive lab</h1>
        <p className="lead">Every visualiser in one place. Each one runs a real implementation of the algorithm in your browser - change the inputs and watch the math respond.</p>
      </header>
      <Lab />
    </article>
  );
}
