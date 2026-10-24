# Machine Learning 1 → 100

**All of classical machine learning in one repository: from collecting raw data to gradient boosting, neural
networks and reinforcement learning. Every algorithm is derived mathematically and implemented from scratch.
It ships with an interactive local web app and 5 end-to-end projects on real datasets.**

- **58 lessons in 13 parts.** Each lesson has the background, why you need it, use cases, the maths, a workflow
  diagram, the from-scratch code, pros and cons, and a runnable demo.
- **`mlscratch/`** is a NumPy-only ML library you can read line by line: about 3,500 lines and 60+ algorithms, all
  tested against scikit-learn.
- **`cpp/`** holds the core algorithms again as header-only C++17, with CMake and tests.
- **A local web app** with 16 interactive visualisers, rendered maths, diagrams, an algorithm chooser, progress
  tracking and **Run** buttons that execute the real Python code and show its plots.
- **5 projects** from beginner to advanced, on data from Kaggle, UCI and the original sources.

```
data collection → wrangling → EDA → feature engineering → feature selection/extraction → evaluation
      → regression → classification → ensembles → unsupervised → neural networks → reinforcement learning → projects
```

---

## Start: one command

```bash
git clone https://github.com/KushalranaAI/machine-learning-1-to-100.git
cd machine-learning-1-to-100
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
python start.py
```

`start.py` checks Python (3.10+) and Node.js (20.9+). It installs the Python and npm packages if they're missing,
builds the web app once, starts it at **http://127.0.0.1:3000** and opens your browser.

| Option | What it does |
|---|---|
| `python start.py --dev` | hot-reloading dev server, for when you edit the web app itself |
| `python start.py --port 4000 --no-browser` | different port, no browser tab |
| `python start.py --skip-install` / `--reinstall` | skip or force the dependency step |

The app only listens on `127.0.0.1`, because its Run buttons execute the course scripts on your machine.

### Using the app

- **Guided lessons.** Each lesson is a short journey: Overview → Learn → Workflow → Interact → Code → Run → Practice → Notes.
  Mark lessons complete, keep notes, and pick up where you left off from the dashboard. **My progress** shows your streak,
  activity and milestones, and can export or import a backup.
- **Search anything** with `Ctrl K` (`Cmd K` on macOS): lessons, algorithms, projects.
- **AI tutor (optional, bring your own).** Press `Ctrl /` or **Ask AI**. Highlight any text, press **Explain** on a code block,
  or use **Ask about this** on a visual, an algorithm or a project step. The tutor sees the page and step you are on, and can
  write a self-check quiz from the lesson.
  - **Local and private:** [Ollama](https://ollama.com) (`ollama pull llama3.2`), LM Studio, llama.cpp or any OpenAI-compatible server.
  - **Cloud:** Claude, OpenAI or Gemini with your own API key.
  - **Azure OpenAI** (your endpoint, deployment name and API key) is offered only when the app runs on `localhost` / `127.0.0.1`,
    and is hidden on any other address, for example a hosted copy of the app.
  - **Your keys stay yours.** Settings, keys and chats are stored only in your browser (`localStorage`, or `sessionStorage` if you
    choose "this session only") and are sent straight from the browser to the provider you pick. The app's server never sees them
    and nothing is written to this repository. Don't save keys on a shared computer.
  - For LM Studio, enable CORS in its server settings. Ollama works out of the box on `localhost`.

**Without the web app:** everything is plain Python. Run `pip install -r requirements.txt && pip install -e .`, then
`python curriculum/07-regression/01-linear-regression/demo.py`. Figures are saved to `outputs/` next to the script.

---

## Curriculum

| Part | Lessons |
|---|---|
| [00 Math Foundations](curriculum/00-math-foundations) | Linear algebra · Calculus & gradients · Probability & statistics · Optimisation & gradient descent |
| [01 Data Collection](curriculum/01-data-collection) | Sources & file formats · APIs · Web scraping (with robots.txt) · Databases & SQL · Public datasets & the Kaggle API |
| [02 Data Handling & Wrangling](curriculum/02-data-wrangling) | pandas essentials · Cleaning & missing values · Outliers · Reshaping & merging · Data validation |
| [03 Exploratory Data Analysis](curriculum/03-eda) | Descriptive statistics & visualisation · Correlation & hypothesis tests |
| [04 Feature Engineering](curriculum/04-feature-engineering) | Scaling · Encoding categoricals · Transformations & binning · Feature creation & leakage · Text features (BoW, TF-IDF) |
| [05 Feature Selection & Extraction](curriculum/05-feature-selection-extraction) | Filter methods · Wrapper & embedded methods · PCA · LDA, kernel PCA & t-SNE |
| [06 Model Evaluation & Selection](curriculum/06-model-evaluation) | Splits & cross-validation · Regression metrics · Classification metrics · Bias-variance · Hyper-parameter tuning · Imbalanced data |
| [07 Regression](curriculum/07-regression) | Linear regression (OLS) · Polynomial regression · Ridge, Lasso & Elastic Net |
| [08 Classification](curriculum/08-classification) | Logistic regression · Softmax regression · k-NN · Naive Bayes · SVMs (SMO + kernels) · Decision trees (CART) |
| [09 Ensemble Methods](curriculum/09-ensembles) | Bagging & random forests · AdaBoost · Gradient boosting · XGBoost (second-order boosting) |
| [10 Unsupervised Learning](curriculum/10-unsupervised) | k-means · Hierarchical clustering · DBSCAN · Gaussian mixtures & EM · Anomaly detection · Association rules |
| [11 Neural Networks](curriculum/11-neural-networks) | Perceptron · MLP & backpropagation · Training deep networks · CNNs · RNNs |
| [12 Reinforcement Learning](curriculum/12-reinforcement-learning) | Multi-armed bandits · MDPs & dynamic programming · Q-learning & SARSA |

Every lesson folder contains `README.md` (the lesson, with frontmatter the web app turns into info cards) and
`demo.py` (it runs the algorithm on data, prints the numbers the lesson refers to, and saves plots).

## Projects

| # | Project | Level | Data | Algorithms | Result |
|---|---|---|---|---|---|
| 01 | [California House Prices](projects/01-house-prices) | Beginner | Kaggle / 1990 census, 20k rows | OLS, Ridge, kNN, trees, random forest, gradient boosting | test R² 0.84, median error 10.7% |
| 02 | [Titanic Survival](projects/02-titanic-survival) | Beginner+ | Kaggle competition, 891 rows | 8 classifiers, grid search | CV 0.837, test accuracy 0.82 |
| 03 | [Customer Segmentation](projects/03-customer-segmentation) | Intermediate | Kaggle mall customers | k-means, Ward, GMM, DBSCAN, PCA | 5 segments, bootstrap ARI 0.975 |
| 04 | [SMS Spam Detector](projects/04-spam-detector) | Intermediate+ | UCI SMS Spam Collection | naive Bayes, logistic regression, linear SVM, TF-IDF | test F1 0.96 at ~97% precision |
| 05 | [Handwritten Digit Recognizer](projects/05-digit-recognizer) | Advanced | MNIST, 70k images | PCA, kNN, MLP, CNN, all from scratch | CNN 98.4% test accuracy |

Each project has a step-by-step guide in its README (and in the web app). Every `download_data.py` tries the
**Kaggle API** first and falls back to the original source or a public mirror. To use Kaggle, create a token at
kaggle.com → Settings → *Create New Token* and save it as `~/.kaggle/kaggle.json`.

---

## The from-scratch library: `mlscratch`

A scikit-learn-style API (`fit`, `predict`, `predict_proba`, `transform`), written for reading, not for speed.

| Module | Inside |
|---|---|
| `linear_model` | LinearRegression (normal equation + GD), Ridge, Lasso, ElasticNet (coordinate descent), LogisticRegression, SoftmaxRegression, Perceptron |
| `neighbors` · `naive_bayes` · `svm` | k-NN, Gaussian/Multinomial/Bernoulli NB, LinearSVC, kernel SVC trained with SMO |
| `tree` · `ensemble` | CART (Gini/entropy/MSE), bagging, random forests, AdaBoost (SAMME), gradient boosting, an XGBoost-style booster |
| `cluster` · `decomposition` · `anomaly` | k-means++, agglomerative (Lance-Williams), DBSCAN, GMM (EM) · PCA, TruncatedSVD, kernel PCA, LDA, t-SNE · isolation forest, Mahalanobis |
| `neural_network` · `optim` | Dense, activations, dropout, batch norm, Conv2D (im2col), max-pool, RNN, gradient checking · SGD, momentum, RMSProp, Adam |
| `preprocessing` · `feature_selection` · `text` | scalers, power transform, encoders (incl. target encoding), imputers (incl. KNN), polynomial features, pipelines · filter/wrapper/embedded selection · CountVectorizer, TF-IDF |
| `model_selection` · `metrics` | splits, (stratified) k-fold, grid/random search, learning curves · regression, classification, ROC/PR, silhouette |
| `rl` · `association` · `download` | grid worlds, value/policy iteration, Q-learning/SARSA, bandits · Apriori · robust Kaggle/URL dataset fetching |

## C++ implementations: `cpp/`

Linear regression (Cholesky normal equation + GD), logistic regression, k-NN, Gaussian naive Bayes, a CART tree,
k-means++, PCA by power iteration, and an MLP with backprop. They're header-only C++17 with no dependencies.

```bash
cd cpp && cmake -S . -B build && cmake --build build -j && ctest --test-dir build && ./build/kmeans
```

## Tests

```bash
pip install -r requirements.txt && pip install -e .
pytest -q        # 104 tests: every algorithm vs scikit-learn, plus lesson/project metadata checks
```

## Repository layout

```
start.py          one-command launcher for the web app
curriculum/       13 parts × lessons: README.md (theory + maths + workflow) and demo.py
projects/         5 projects: README.md (guide) + download_data.py + analysis/training scripts
mlscratch/        the from-scratch NumPy library used by every lesson and project
cpp/              C++17 header-only implementations, examples, tests (CMake)
web/              Next.js app: renders the course, interactive widgets, runs scripts locally
data/             small bundled datasets (iris, a deliberately messy customer table)
tests/            pytest suite
```

## How to study
1. Follow the parts in order (the roadmap page shows your progress), or jump in via the **Algorithm map**.
2. For each lesson: read it, play with its widget, run the demo, then open the **Code** tab and trace the maths in the implementation.
3. After parts 06, 09 and 11, do the matching projects. Then do their "Extend it" exercises on a dataset you care about.

## License
MIT. See [LICENSE](LICENSE). The datasets keep their original licenses (listed in each project's README).
