# Security policy

## Reporting a vulnerability

Please **don't open a public issue** for a security problem. Use **Security → Report a vulnerability** on this
repository's GitHub page, which opens a private report only the maintainer can see. Include what you found, how to
reproduce it and, if you can, what you think the impact is. You'll get an answer as soon as the maintainer can.

## What this project does that is worth knowing

This is a learning app that you run **on your own computer**. Two features deserve care:

- **Run buttons execute code.** The web app's `/api/run` route runs the lesson and project scripts with your local Python.
  It only runs `.py` files inside `curriculum/` and `projects/`, only accepts requests addressed to `localhost`,
  `127.0.0.1` or `[::1]`, and rejects cross-origin requests. `start.py` listens on `127.0.0.1` only. Don't expose the app
  to a network or the internet, and don't start it with `--hostname 0.0.0.0`.
- **The AI tutor uses your own API keys.** Keys, settings and chats are stored only in your browser (`localStorage`,
  or `sessionStorage` if you choose "this session only") and are sent directly from the browser to the provider you
  pick. The app's server never receives them, and nothing is written to this repository. Anyone who can use your
  browser profile can read them, so don't save keys on a shared computer, and prefer keys with a spending limit.
  Azure OpenAI is offered only when the app runs on `localhost`.

## For contributors

- Never commit API keys, tokens, `.env` files or `kaggle.json`. `.gitignore` covers the common cases, but check your diff.
- If a key is ever committed, revoke it at the provider first; removing it from git history is not enough.
- `main` is protected: changes arrive through reviewed pull requests, and CI (`python`, `web`, `cpp`) must pass.
