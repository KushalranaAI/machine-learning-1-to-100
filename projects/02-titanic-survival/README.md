---
title: "Titanic Survival"
level: Beginner+
summary: "Kaggle's classic: turn messy passenger records (names, tickets, cabins) into features and compare 8 classifiers from scratch with stratified cross-validation."
problem: |
  Given what was known about each passenger (class, sex, age, family aboard, fare, cabin, port), predict who
  survived the sinking. This is a **binary classification** problem on a small (891 rows), messy dataset. Most of the
  work is feature engineering from text columns, plus a validation protocol you can trust on so little data.
dataset:
  name: "Titanic - Machine Learning from Disaster"
  source: "Kaggle competition · titanic"
  url: "https://www.kaggle.com/competitions/titanic"
  kaggle: "competitions/titanic (accept the rules once on the website)"
  license: "Public historical record. Mirrors: datasciencedojo, pandas-dev on GitHub"
  size: "891 labelled passengers × 12 columns · 60 KB"
algorithms: [Logistic Regression, kNN, Gaussian Naive Bayes, SVM (RBF), Decision Tree, Random Forest, AdaBoost, Gradient Boosting]
skills:
  - Regex feature extraction (titles from names, decks from cabins)
  - Group-aware imputation (age by title)
  - Missingness as a feature
  - Leak-free cross-validation (features refit inside each fold)
  - Stratified splits, grid search, ROC curves and decision thresholds
  - Interpreting logistic-regression coefficients
lessons:
  - 02-data-wrangling/02-data-cleaning
  - 04-feature-engineering/02-encoding-categorical
  - 04-feature-engineering/04-feature-creation
  - 06-model-evaluation/01-train-test-cross-validation
  - 06-model-evaluation/03-classification-metrics
  - 06-model-evaluation/05-hyperparameter-tuning
  - 08-classification/01-logistic-regression
  - 08-classification/04-naive-bayes
  - 08-classification/05-support-vector-machines
  - 08-classification/06-decision-trees
  - 09-ensembles/01-bagging-random-forest
  - 09-ensembles/02-adaboost
  - 09-ensembles/03-gradient-boosting
steps:
  - title: "1 · Download"
    script: download_data.py
    description: |
      With a Kaggle token, this downloads the competition's `train.csv` (and `test.csv` for a leaderboard submission).
      Without one, it downloads the same `train.csv` from a GitHub mirror.
  - title: "2 · Explore"
    script: explore.py
    description: |
      38% survived. **Sex** and **class** dominate: 97% of first-class women survived, but only 14% of third-class men.
      `Age` is missing for 177 passengers and `Cabin` for 687. The title hidden in each name ("Master" = boy, median
      age 3.5) is the best way to fill in age.
  - title: "3 · Feature engineering (features.py)"
    description: |
      - `Title` from the name by regex. Rare titles are grouped together.
      - `FamilySize`, `IsAlone`, `TicketGroup` (people sharing a ticket travelled together), `FarePerPerson`
      - `HasCabin` and `Deck`: here the *missingness* itself predicts survival (the crew recorded cabins mostly for first class)
      - `Age` imputed with the median of the passenger's title, plus an `AgeMissing` flag
      - one-hot `Sex`, `Embarked`, `Title`, `Deck`, `Pclass`; standardised numerics

      `FeaturePipeline` wraps features and model, so every CV fold relearns these statistics. That means no leakage.
  - title: "4 · Model comparison, tuning, test"
    script: train.py
    description: |
      Stratified 5-fold CV for 8 models, then a grid search for the SVM and the forest. The best model is scored once
      on a stratified 20% hold-out set. If Kaggle's `test.csv` is present, the script also writes `submission.csv`.
      It takes about 3 minutes.
workflow: |
  flowchart LR
    A[train.csv] --> B[Stratified 80/20 split]
    B --> C[Parse text: Title, Deck, TicketGroup]
    C --> D[Impute by title, one-hot, scale<br/>inside each CV fold]
    D --> E[8 models · 5-fold CV]
    E --> F[Grid search SVM + RF]
    F --> G[Best → test once]
    G --> H[ROC · threshold · coefficients]
---

# Titanic: feature engineering beats model choice

## Results (seeded)

| Model | 5-fold CV accuracy |
|---|---|
| Gaussian naive Bayes | 0.740 ± 0.028 |
| Decision tree (depth 4) | 0.806 ± 0.029 |
| kNN (k = 9) | 0.810 ± 0.031 |
| Random forest | 0.826 ± 0.022 |
| AdaBoost | 0.829 ± 0.028 |
| Logistic regression | 0.830 ± 0.030 |
| SVM (RBF) | 0.830 ± 0.021 |
| Gradient boosting | 0.831 ± 0.023 |
| **Random forest, tuned (depth 6, leaf 3)** | **0.837** |

**Test (179 passengers, once):** accuracy 0.816, F1 0.75. Gradient boosting's ROC-AUC is 0.868.

## What to notice
- **Seven of eight models land within one standard deviation of each other** (0.81-0.84). With 712 training rows, the
  feature set matters far more than the algorithm. A ±0.03 CV spread also means differences of 0.005 are noise.
- **Naive Bayes is last** because its independence assumption is badly violated: `Sex`, `Title=Mr` and `Title=Mrs`
  say almost the same thing, so NB counts that evidence three times.
- **Logistic regression is readable:** being male (−0.45), a "Mr" (−0.43) and a high `SibSp` (−0.56) lower your odds;
  "Master" (+0.42, young boys) and a higher fare raise them. "Women and children first" is right there in the weights.
- **Thresholds are a business choice:** at 0.3 you catch 81% of survivors at 73% precision; at 0.7 you catch 65% at 85% precision.

## Extend it
1. Add `Surname` + `TicketGroup` "family survival" features. Top Kaggle notebooks use these. Watch for leakage across folds!
2. Stack the logistic regression, SVM and boosting models with a meta logistic regression.
3. Submit `submission.csv` to Kaggle and compare the public leaderboard with your CV estimate.
