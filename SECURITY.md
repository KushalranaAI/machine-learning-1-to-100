# Security policy

## Reporting a vulnerability

Please **don't open a public issue** for a security problem. Use **Security → Report a vulnerability** on this
repository's GitHub page, which opens a private report only the maintainer can see. Include what you found, how to
reproduce it and, if you can, what you think the impact is. You'll get an answer as soon as the maintainer can.

## What this project does that is worth knowing

This is a learning app that you run **on your own computer**. A few things deserve care:

- **Run buttons execute code.** The web app's `/api/run` route runs the lesson and project scripts with your local Python.
  It only runs `.py` files inside `curriculum/` and `projects/`, only accepts requests addressed to `localhost`,
  `127.0.0.1` or `[::1]`, and rejects cross-origin requests. `start.py` listens on `127.0.0.1` only. Don't expose the app
  to a network or the internet, and don't start it with `--hostname 0.0.0.0`.
- **The hosted copy is static.** The GitHub Pages site is plain files: there is no server, nothing executes on the
  host, and the `/api/run` route does not exist there. Visitors' progress, notes and AI keys stay in their own browser.
- **The AI tutor uses your own API keys, in one of three ways.**
  1. *Saved in the browser, encrypted.* API keys, endpoints and regions are stored only as an AES-256-GCM envelope
     (`web/lib/ai/vault.ts`) in `localStorage`, or `sessionStorage` for "this session only". With **device** protection the key is
     a non-extractable WebCrypto key kept in IndexedDB; with a **passphrase** it is derived with PBKDF2-SHA256 (600,000 rounds)
     and kept only in memory. Tampered data fails to decrypt, and credentials are never written unencrypted (on a
     non-HTTPS, non-localhost page they stay in memory only).
  2. *In `.env` on your computer.* The local server (`web/server/ai-relay.ts`) attaches the key. It only calls fixed provider URLs,
     only answers requests from the local page (Host must be localhost, any Origin must match, POSTs need a custom header),
     reports configured providers as booleans only, and strips echoed keys from error messages. The keys are not in the
     browser, not in the built JavaScript, and not in the repository (`.env` is git-ignored).
  3. *Direct to the provider.* Browser requests go straight to the provider you chose; the app's server never receives them.

  What this does not stop: with device protection, script already running in the page (for example through a
  cross-site-scripting bug) or someone using your unlocked browser can still use a saved key, and a passphrase cannot protect a
  key from script running while you are using the app. Prefer keys with a spending limit. Azure OpenAI is offered only on
  `localhost`.
- **Content-Security-Policy.** Pages limit where scripts load from and where data can be sent: only this app, the supported AI
  providers and servers on this computer, so an injected script cannot send a saved key to an unknown site. The hosted copy
  also allows only its own inline scripts, by hash. The local app has to allow inline scripts because Next.js's are generated
  per request, so for it the connection limit is the protection that matters.

## For contributors

- Never commit API keys, tokens, `.env` files or `kaggle.json`. `.gitignore` covers the common cases, but check your diff.
- If a key is ever committed, revoke it at the provider first; removing it from git history is not enough.
- `main` is protected: changes arrive through reviewed pull requests, and CI (`python`, `web`, `cpp`) must pass.
