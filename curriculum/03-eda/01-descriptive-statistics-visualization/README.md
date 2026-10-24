---
title: Descriptive Statistics & Visualisation
order: 1
summary: Summarise and plot data - centre, spread, shape, histograms, box plots, scatter plots, grouped comparisons and correlation heatmaps - and learn what to look for.
level: beginner
minutes: 40
why: Summary numbers can hide everything (Anscombe's quartet has identical means, variances and correlations but completely different shapes). Plots reveal skew, outliers, clusters, non-linearity and leakage that a model would otherwise learn blindly.
background: John Tukey's "Exploratory Data Analysis" (1977) made looking at data a discipline of its own. Anscombe's quartet (1973) remains the classic warning.
use_cases:
  - First look at every new dataset
  - Choosing transformations (log for skewed income)
  - Spotting class imbalance and leakage
  - Communicating findings to stakeholders
where_used:
  - 01-house-prices
  - 02-titanic-survival
  - 03-customer-segmentation
demo: demo.py
workflow: |
  flowchart LR
    A[describe: centre & spread] --> B[histograms: shape, skew]
    B --> C[box plots by group]
    C --> D[scatter: relationships]
    D --> E[correlation heatmap]
    E --> F[write down findings + feature ideas]
---

## Centre, spread and shape
| Statistic | Formula | Robust? |
|-----------|---------|---------|
| mean | $\bar x = \frac1n\sum x_i$ | ✗ pulled by outliers |
| median | middle value | ✓ |
| variance / std | $s^2 = \frac{1}{n-1}\sum (x_i-\bar x)^2$ | ✗ |
| IQR | $Q_3 - Q_1$ | ✓ |
| skewness | $\frac{1}{n}\sum\left(\frac{x_i-\bar x}{s}\right)^3$ | ✗ |
| kurtosis | $\frac{1}{n}\sum\left(\frac{x_i-\bar x}{s}\right)^4 - 3$ (tails) | ✗ |

Mean > median ⇒ right skew (income, prices, counts). These usually benefit from a **log transform**.

## Which plot for which question
| Question | Plot |
|----------|------|
| How is one numeric variable distributed? | histogram / KDE, box plot |
| How often does each category occur? | bar chart of `value_counts()` |
| Does a numeric feature differ by class? | box/violin plot per class, overlaid histograms |
| Are two numeric variables related? | scatter plot (add transparency for many points) |
| How do all numeric variables relate? | correlation heatmap, pair plot |
| Does a category relate to the target? | target rate per category (bar), crosstab |

## Anscombe's quartet
Four datasets with the same mean, variance, correlation (0.816) and regression line - and four completely different
pictures (linear, curved, one outlier, one leverage point). **Always plot.** The demo recreates them.

## A checklist for every dataset
- Target: balanced? skewed? (regression → consider log)
- Each feature: type, missing %, unique values, range, skew, impossible values
- Feature vs target: which features separate the classes / correlate with the target?
- Features vs features: redundant (|r| > 0.9) groups?
- Anything *too* predictive? (an ID or a post-outcome column → **leakage**)

## Exercises
1. Run the demo; describe each Anscombe dataset in one sentence.
2. Which iris feature separates the species best? Back it up with a plot.
3. For `clean_customers.csv`, plot income on a linear and a log axis. Which is more informative?
