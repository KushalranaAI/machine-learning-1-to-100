# Contributing to Machine Learning 1 → 100

Thanks for helping. Anyone can read, clone, fork and run this project. Changes reach `main` only through a pull
request that the maintainer reviews and merges, so you never need (or get) write access to this repository.

## The short version

1. **Fork** this repository (the Fork button at the top right of GitHub).
2. **Clone your fork** and create a branch:
   ```bash
   git clone https://github.com/<your-username>/machine-learning-1-to-100.git
   cd machine-learning-1-to-100
   git switch -c my-change
   ```
3. **Make your change**, then check it (see below).
4. **Push the branch to your fork** and open a **pull request** against `main` of this repository.
5. A maintainer reviews it. Automated checks run first. The first time you contribute, a maintainer has to approve
   those checks before they start; that is normal.

You can't push to this repository directly, and `main` is protected, so don't worry about breaking anything.

## Run it locally

You need Python 3.10+ and Node.js 20.9+ (CI uses Python 3.12 and Node 22).

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
python start.py            # installs what's missing, builds the web app once, opens http://127.0.0.1:3000
python start.py --dev      # hot-reloading dev server, for when you edit the web app itself
```

## Check your change before opening the pull request

CI runs three independent checks. Run the ones that match what you touched:

```bash
python -m pytest -q                                # lessons, projects and every algorithm vs scikit-learn
cd web && npm run verify                           # web app: type check + production build
cmake -S cpp -B cpp/build && cmake --build cpp/build && ctest --test-dir cpp/build --output-on-failure   # C++
```

If these pass on your machine they will pass in CI.

## What to work on

| Kind of change | Where |
|---|---|
| A new lesson, or a fix to one | `curriculum/<part>/<lesson>/README.md` (the frontmatter drives the web app's cards) |
| A new algorithm | `mlscratch/`, with a test against scikit-learn in `tests/` |
| A project | `projects/<nn-name>/` |
| The C++ versions | `cpp/` |
| The web app | `web/` (Next.js; read `web/AGENTS.md` first) |

Good first contributions: fix a typo or an unclear sentence in a lesson, improve an explanation, add a missing
exercise, or report something that confused you.

## Guidelines

- **Keep it original.** Learn from any source you like, but write lessons and projects in your own words with your own
  examples. Only submit work you have the right to share; contributions are released under the project's [MIT license](LICENSE).
- **One topic per pull request.** Small changes are reviewed faster.
- **Tests for algorithm changes.** New or changed code in `mlscratch/` needs a test that compares it with a trusted reference.
- **Quote YAML text that contains a colon.** In lesson frontmatter write `- "Fair protocol: train / test"`, not
  `- Fair protocol: train / test`. Otherwise YAML reads it as a key and value; a test checks this.
- **Escape currency in Markdown.** Write `\$40k`, because `$...$` is rendered as math.
- **Never commit secrets.** API keys belong only in your own browser (the AI tutor stores them there) or in your
  own environment. Don't add keys, tokens, `.env` files or `kaggle.json`.
- **Don't commit data or generated output.** Datasets are downloaded by each project's `download_data.py`, and plots go to `outputs/`.
- **Match the surrounding code.** Same naming, comment density and style as the file you are editing.
- **Be kind.** Disagree with ideas, not people. Beginners ask the best questions.

## Reporting problems

- **A bug or an unclear lesson:** open an issue with what you did, what you expected and what happened. A screenshot helps a lot.
- **A security problem:** please don't open a public issue. Use the repository's **Security → Report a vulnerability**
  button so it can be fixed before it is public. See [SECURITY.md](SECURITY.md).

## How pull requests are handled

- Only the maintainer can merge into `main`. Contributors, including collaborators, work through pull requests.
- The `python`, `web` and `cpp` checks must pass, and the maintainer must approve (see `.github/CODEOWNERS`).
- Pull requests from first-time contributors need a maintainer to approve the automated checks before they run.
- The repository's protection rules live in `.github/rulesets/protect-main.json`.
