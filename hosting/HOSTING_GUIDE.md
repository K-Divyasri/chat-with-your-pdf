# Publishing Chat With Your PDF

This one is different from a command-line project — you built an actual **web app**
(`app.py`, a Streamlit interface). So "hosting" means two things here, and you want
both:

1. **Get the code on GitHub** — a clean repo with a README a recruiter can open, and
   CI that proves the tests pass on every push.
2. **Put the app on the internet** — a live URL where someone can upload a PDF, ask a
   question, and watch the retrieval work. That live link is the thing that makes this
   project *land* in an interview.

The good news: this app is cheap and easy to host, because of one design decision.
There's **no database to run**. The vector index is built in memory from whatever PDFs
the visitor uploads, right there at request time, and thrown away when they leave. No
server to keep warm, no DB to pay for, no data to store. That's a real selling point —
it's why a project like this fits on a free hosting tier. More on that below.

And by default the app runs **fully offline** — the hashing embedder plus extractive
answers, no API key needed. So the deployed demo works for anyone, immediately, and
costs you nothing. Real-LLM answers are an optional switch you can turn on with a key.

---

## The layout you're working with

Your project folder looks like this:

```
08-chat-with-your-pdf/          <- this whole folder becomes your GitHub repo
├── build_from_scratch/         <- the real app lives here
│   ├── app.py                  <- the Streamlit web app
│   ├── pdf_chat/               <- the package it imports (loading, chunking, ...)
│   ├── requirements.txt
│   ├── tests/                  <- 25 offline tests
│   └── generate_data.py
├── hosting/                    <- you are here
├── data/                       <- sample PDFs (after you run generate_data.py)
├── generate_data.py
└── README.md
```

Two things follow from this:

- **GitHub gets the whole `08-chat-with-your-pdf/` folder.** The CI workflow and the
  root README are written for that. The app *code* the app needs is the
  `build_from_scratch/` subfolder.
- **The live app is deployed separately** (Path A or Path B below). Both point at
  `build_from_scratch/app.py`.

---

## Step 0 — Get the code on GitHub

If you did Project 1, this is the same dance. Skim it; the new bits are flagged.

### Install Git and tell it who you are (once per machine)

If you already have Git working, skip this. Otherwise: download from
<https://git-scm.com/download/win>, run the installer clicking Next through the
defaults, then open a **new** PowerShell window and check:

```powershell
git --version
```

Then stamp your identity onto commits (use the same email as your GitHub account):

```powershell
git config --global user.name "Your Name"
git config --global user.email "mathuransada@gmail.com"
```

### Know what must NOT go in the repo

The project already ships a `.gitignore` inside `build_from_scratch/`. Open it and
confirm it lists at least these — they're the ones that matter:

```
.env
.index/
*.index/
.venv/
__pycache__/
.pytest_cache/
```

Why each one is excluded:

- **`.env`** — this is the important one. If you ever add a Gemini API key for real-LLM
  answers, it goes in `.env`. A key is a password. Commit it once and it's on the public
  internet **forever** (Git keeps the whole history, and bots scrape GitHub for leaked
  keys within minutes). Someone then runs up a bill on your account. So `.env` **never**
  gets committed. The repo ships `.env.example` instead — variable names, blank values —
  which is safe and *is* meant to be committed.
- **`.index/`** — the saved vector index the CLI writes with `--index .index`. It's
  generated output, rebuilt from the PDFs any time. No reason to store it in Git.
- **`.venv/`, `__pycache__/`, `.pytest_cache/`** — your virtual environment and the junk
  Python/pytest create as they run. Machine-specific; other people rebuild the venv from
  `requirements.txt`.

Rule of thumb: **source code, config, docs, and the sample data-generator go in.
Secrets, the built index, and machine junk stay out.**

### Make the repo and push

Run these from the **project root** — the `08-chat-with-your-pdf/` folder, the one with
`build_from_scratch/` and this `hosting/` folder inside it. (This differs from Project 1,
where the repo was just the inner folder. Here you publish the whole learning kit.)

```powershell
cd ai\08-chat-with-your-pdf
git init
git add .
git commit -m "Initial commit: Chat With Your PDF (classic RAG)"
```

Now the single most important check in this whole guide:

```powershell
git status
```

You want `nothing to commit, working tree clean`, and you must **not** see `.env`
anywhere. Double-check with:

```powershell
git ls-files | Select-String ".env"
```

You should see `build_from_scratch/.env.example` and nothing else. If a bare `.env`
shows up, you committed a secret — jump to *Committed .env by accident* at the bottom
and fix it before you push.

Then make an **empty** repo on github.com (the **+** menu, top-right → **New
repository**), name it `chat-with-your-pdf`, leave it **Public**, and do **not** tick
"Add a README / .gitignore / license" (an empty repo avoids a first-push collision).
Copy the repo URL it shows you, then back in PowerShell:

```powershell
git branch -M main
git remote add origin https://github.com/YOURNAME/chat-with-your-pdf.git
git push -u origin main
```

The first push opens a browser to sign in to GitHub. If it asks for a *password* typed
into the terminal, that won't work — GitHub turned off password auth years ago. Use the
browser sign-in it offers, or install the GitHub CLI (<https://cli.github.com>) and run
`gh auth login` once. Refresh the repo page and your files are there.

### Add CI so the tests run on every push

CI proves your tests pass on a clean machine, not just your laptop, every single push —
and GitHub shows a green checkmark that recruiters notice. This `hosting/` folder ships
a ready workflow at `github_actions/ci.yml`. GitHub only runs workflows that live under
`.github/workflows/`, so copy it there. From the **project root**:

```powershell
mkdir .github\workflows
copy hosting\github_actions\ci.yml .github\workflows\ci.yml
git add .github\workflows\ci.yml
git commit -m "Add GitHub Actions CI to run the offline tests on every push"
git push
```

Open the repo's **Actions** tab to watch it run: checkout → install Python → install
deps → `pytest`. It's **keyless** — every test runs offline, so nothing needs a secret.
Green means all 25 passed on GitHub's machine. If it goes red, click the failed step and
read the log bottom-up; a missing dependency in `requirements.txt` is the usual cause.

Once it's green, grab the status badge (the workflow's Actions page has a `...` menu →
**Create status badge**) and paste the markdown at the top of your root `README.md`.

---

## The app is deployed separately — pick a path

The GitHub repo above is your *code*. To get a *live app*, use one of the two paths
below. **Path A (Hugging Face Spaces) is the one I'd pick** — it's built for exactly this
and needs no card on file. Path B (Streamlit Community Cloud) is just as free and wires
straight to the GitHub repo you already pushed; use it if you prefer that.

Either way, remember the app runs **offline by default**, so the moment it builds, it
works for anyone — no key required.

---

## Path A (recommended) — Hugging Face Spaces

A "Space" is a free, always-on little web app hosted by Hugging Face. It speaks Streamlit
natively, so there's almost nothing to configure. Official docs (worth a skim):
<https://huggingface.co/docs/hub/en/spaces-sdks-streamlit>.

### A1. Create the Space

1. Make a free account at <https://huggingface.co>.
2. Top-right, your avatar → **New Space**.
3. **Space name:** `chat-with-your-pdf`. **License:** whatever you like (MIT is common).
4. **Space SDK:** pick **Streamlit**. (This is the important choice — it tells Hugging
   Face to look for an `app.py` and run `streamlit run` on it for you.)
5. Leave hardware on the free **CPU basic** tier. This app is light; it doesn't need a
   GPU. Click **Create Space**.

### A2. Add the app files

A Space is itself a Git repo. It expects the app at its **root** — an `app.py`, a
`requirements.txt`, and any code they import, all at the top level. Our files live inside
`build_from_scratch/`, so you upload *their contents* to the Space root. The three things
the Space needs:

- `app.py`               — the Streamlit app
- `pdf_chat/`            — the package `app.py` imports (the whole folder)
- `requirements.txt`     — so the Space installs numpy, pypdf, streamlit

The easiest way for a beginner is the web uploader: on your Space page, the **Files**
tab → **Add file** → **Upload files**. Drag in `app.py`, `requirements.txt`, and the
entire `pdf_chat` folder (from your `build_from_scratch/`). Commit.

> You do **not** need to upload `tests/`, `data/`, `.index/`, `.venv/`, or
> `generate_data.py` — the app doesn't use them. The visitor supplies their own PDFs by
> uploading, so there's nothing to preload.

Once you commit, the Space starts building. Watch the **Logs** — it installs
`requirements.txt`, then launches Streamlit. A minute or two later you have a live URL
like `https://huggingface.co/spaces/YOURNAME/chat-with-your-pdf`. Open it: it should say
"Upload a PDF to begin." Run `python generate_data.py` locally to get a sample PDF from
`data\` and upload that to try it.

### A3. (Optional) Turn on real-LLM answers with a Secret

The public demo is fine offline — leave it that way and it spends nothing. But if you
want the "Use a real LLM" checkbox to actually work on the hosted app, it needs a Gemini
key. **Never put the key in the repo.** Spaces has a proper vault for it:

1. On the Space, go to **Settings** → **Variables and secrets** → **New secret**.
2. Name it `GEMINI_API_KEY`, paste your key (see *Getting a free Gemini key* below) as
   the value, save.
3. The Space restarts with the key available as an environment variable. LiteLLM reads
   `GEMINI_API_KEY` from the environment, so the checkbox now returns live answers.
   (You'll also need `litellm` installed — add `litellm>=1.40` to the Space's
   `requirements.txt`; it's commented out in the offline default.)

A secret set this way is stored encrypted and is never shown in the repo or the logs.
That's the whole point — the key lives in the vault, not in your code.

---

## Path B — Streamlit Community Cloud

Streamlit's own free host. It connects straight to the GitHub repo you pushed in Step 0
and redeploys whenever you push. Home page: <https://streamlit.io/cloud>.

1. Go to <https://streamlit.io/cloud> and sign in with your GitHub account, granting it
   read access to your repos.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. **Repository:** `YOURNAME/chat-with-your-pdf`. **Branch:** `main`.
4. **Main file path:** this is the key field — point it at **`build_from_scratch/app.py`**
   (not just `app.py`, because the app lives in the subfolder).
5. Click **Deploy**. It reads `requirements.txt` (it sits right next to `app.py` in
   `build_from_scratch/`, where Streamlit Cloud looks), installs the deps, and launches.
   A minute later you have a public `*.streamlit.app` URL.

If the build ever complains it can't find dependencies, add a one-line `requirements.txt`
at the **repo root** containing `-r build_from_scratch/requirements.txt` — that just tells
pip to read the real list. Usually you won't need to.

### Secrets on Streamlit Cloud

Same rule: the key never goes in the repo. Streamlit Cloud has a **Secrets** box (a small
TOML editor) under the app's **⋮** menu → **Settings** → **Secrets**. Add:

```toml
GEMINI_API_KEY = "your-key-here"
```

Save; the app restarts. Streamlit exposes these to your app, and since LiteLLM reads
`GEMINI_API_KEY` from the environment it'll pick it up. As with Path A, add
`litellm>=1.40` to `requirements.txt` if you want the real-LLM path to work online. Leave
it off and the offline demo runs free.

---

## Why there's no database to host (and why that's good)

Most "chat with your docs" tutorials make you stand up a vector database — Pinecone,
Weaviate, a hosted Chroma — and that means an account, a connection string, sometimes a
bill. This app deliberately doesn't.

When a visitor uploads PDFs, `app.py` builds the vector index **in memory** for that
session (`Rag(...).ingest(paths)`), answers their questions against it, and discards it
when they close the tab. Nothing is persisted. That's why it drops onto a free tier with
zero infrastructure: there's no server to keep running between requests and no data at
rest to store or secure.

The trade-off is that the index isn't remembered — every session re-uploads and
re-indexes. For a demo that's exactly right. If you *wanted* persistence (upload once,
query forever), the upgrade is a real vector database: `knowledge/10` in the parent folder
shows how the toy `VectorStore` here maps one-to-one onto **Chroma**
(<https://docs.trychroma.com/docs/overview/introduction>). Add `chromadb` and point the
store at a folder, and the index survives restarts. Worth mentioning in an interview as
"here's how I'd productionise it" — but you don't need it to ship the demo.

---

## Getting a free Gemini key (only if you want real answers)

You don't need this for the public demo — it runs offline. But to try the real-LLM path,
either locally or on the hosted app:

1. Go to <https://aistudio.google.com/app/apikey> and sign in with a Google account.
2. Click **Create API key**. Copy it.
3. Locally: copy `build_from_scratch\.env.example` to `.env`, paste the key after
   `GEMINI_API_KEY=`. On a host: put it in the Space Secret / Streamlit Secrets box as
   above — **not** in the repo.

**Cost:** Google AI Studio has a free tier that's generous for personal use, and Gemini
Flash is their cheap, fast model. Because your public app defaults to **offline**, it
never calls the model unless *you* flip the switch — so the deployed demo costs nothing.
The key only ever spends money when you explicitly turn on real answers.

---

## Common Git mistakes (troubleshooting)

**Committed `.env` by accident.** First, treat the key as compromised — go to Google AI
Studio and **delete/rotate it**, because if you pushed, it's already public. Then remove
the file from Git while keeping it on disk:

```powershell
git rm --cached .env
git commit -m "Remove committed .env"
git push
```

`git rm --cached` only stops tracking it going forward — the key still sits in your Git
*history*, which is exactly why you rotate the key rather than trusting the delete.

**`error: failed to push` / push rejected.** The remote has commits your local repo
doesn't — almost always because you let GitHub add a README or license when creating the
repo. Pull and replay on top, then push:

```powershell
git pull origin main --rebase
git push
```

Next time, create the repo completely empty.

**The Space or Cloud build fails on an import.** Read the build **Logs** bottom-up. The
usual cause is a package the app imports that isn't in `requirements.txt`, or (on Path A)
forgetting to upload the whole `pdf_chat/` folder so `from pdf_chat.pipeline import Rag`
can't find it. Confirm all three pieces — `app.py`, `pdf_chat/`, `requirements.txt` — are
at the Space root.

**Authentication fails on push.** GitHub no longer accepts your account password in the
terminal. Easiest fix: install the GitHub CLI (<https://cli.github.com>) and run
`gh auth login`, following the browser prompts. It handles auth for all future Git
commands.
