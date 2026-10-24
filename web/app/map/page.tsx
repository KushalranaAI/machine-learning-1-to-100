import type { Metadata } from "next";
import AlgorithmMap from "@/components/AlgorithmMap";
import Mermaid from "@/components/Mermaid";
import Icon from "@/components/ui/Icon";

export const metadata: Metadata = { title: "Algorithm map" };

const CHOOSER = `flowchart LR
  S([Start: what do you have?]) --> L{Labels?}
  L -->|yes| T{Target type?}
  L -->|no| U{Goal?}
  L -->|rewards from actions| RL[Bandits · dynamic programming · Q-learning]
  T -->|number| R{Need explanations?}
  R -->|yes| R1[Linear / Ridge / Lasso]
  R -->|no, accuracy| R2[Gradient boosting · random forest]
  T -->|category| C{Data type?}
  C -->|tabular| C1{Rows?}
  C1 -->|< 1k| C2[Logistic regression · SVM · kNN · naive Bayes]
  C1 -->|> 1k| C3[Gradient boosting · random forest]
  C -->|text| C4[Naive Bayes / linear models on tf-idf]
  C -->|images / audio| C5[CNN]
  C -->|sequences| C6[RNN]
  U -->|groups| G{Shape?}
  G -->|round, know k| G1[k-means]
  G -->|soft / elliptical| G2[Gaussian mixture]
  G -->|arbitrary + noise| G3[DBSCAN]
  G -->|hierarchy| G4[Agglomerative]
  U -->|compress / visualise| D[PCA · t-SNE]
  U -->|odd points| A[Isolation forest · Mahalanobis]
  U -->|co-occurrence| AR[Apriori rules]`;

export default function MapPage() {
  return (
    <article>
      <header className="page-head">
        <div className="eyebrow"><Icon name="map" size={14} /> Decide</div>
        <h1>Algorithm map</h1>
        <p className="lead">
          Which algorithm should you use? Start with the decision chart, then filter by your constraints. Every card links to the lesson with the full derivation and the from-scratch code.
        </p>
      </header>
      <h2 className="section-title">Decision chart</h2>
      <Mermaid chart={CHOOSER} />
      <p className="muted">
        Rules of thumb, not laws: always compare against a simple baseline with cross-validation (lesson 06). On tabular data, gradient boosting is usually the strongest single model, and logistic/linear regression is the one you can explain.
      </p>
      <h2 className="section-title">Compare every algorithm</h2>
      <AlgorithmMap />
    </article>
  );
}
