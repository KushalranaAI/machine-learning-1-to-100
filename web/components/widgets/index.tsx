"use client";
// Registry of interactive visualisers. A lesson lists the ones it uses in its `widgets:` frontmatter.
import dynamic from "next/dynamic";
import type { ComponentType } from "react";
import type { ModelKind } from "./Classifier2D";

const Classifier2D = dynamic(() => import("./Classifier2D"), { ssr: false, loading: () => <p className="muted">Loading widget…</p> });
const classifier = (initial: ModelKind, defaultData = "blobs", allowSwitch = false) => {
  const C = () => <Classifier2D initial={initial} defaultData={defaultData} allowSwitch={allowSwitch} />;
  C.displayName = `Classifier-${initial}`;
  return C;
};

export const WIDGETS: Record<string, ComponentType> = {
  "gradient-descent": dynamic(() => import("./GradientDescent"), { ssr: false, loading: () => <p className="muted">Loading widget…</p> }),
  "linear-regression": dynamic(() => import("./LinearRegression"), { ssr: false, loading: () => <p className="muted">Loading widget…</p> }),
  "bias-variance": dynamic(() => import("./BiasVariance"), { ssr: false, loading: () => <p className="muted">Loading widget…</p> }),
  "logistic-regression": classifier("logistic"),
  svm: classifier("svm"),
  knn: classifier("knn", "moons"),
  "decision-tree": classifier("tree", "moons"),
  "naive-bayes": classifier("nb"),
  "neural-network": classifier("mlp", "circles"),
  "classifier-playground": classifier("logistic", "moons", true),
  "tree-split": dynamic(() => import("./TreeSplit"), { ssr: false, loading: () => <p className="muted">Loading widget…</p> }),
  kmeans: dynamic(() => import("./KMeans"), { ssr: false, loading: () => <p className="muted">Loading widget…</p> }),
  dbscan: dynamic(() => import("./DBSCAN"), { ssr: false, loading: () => <p className="muted">Loading widget…</p> }),
  pca: dynamic(() => import("./PCA"), { ssr: false, loading: () => <p className="muted">Loading widget…</p> }),
  roc: dynamic(() => import("./ROC"), { ssr: false, loading: () => <p className="muted">Loading widget…</p> }),
  scaling: dynamic(() => import("./Scaling"), { ssr: false, loading: () => <p className="muted">Loading widget…</p> }),
  "q-learning": dynamic(() => import("./GridWorld"), { ssr: false, loading: () => <p className="muted">Loading widget…</p> }),
  boosting: dynamic(() => import("./Boosting"), { ssr: false, loading: () => <p className="muted">Loading widget…</p> }),
  distributions: dynamic(() => import("./Distributions"), { ssr: false, loading: () => <p className="muted">Loading widget…</p> }),
  "matrix-transform": dynamic(() => import("./MatrixTransform"), { ssr: false, loading: () => <p className="muted">Loading widget…</p> }),
  outliers: dynamic(() => import("./Outliers"), { ssr: false, loading: () => <p className="muted">Loading widget…</p> }),
  "cross-validation": dynamic(() => import("./CrossValidation"), { ssr: false, loading: () => <p className="muted">Loading widget…</p> }),
};
