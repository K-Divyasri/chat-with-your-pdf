# Deploy checklist — Chat With Your PDF

This is the project's "definition of done." Walk it top to bottom. Don't tick a box you
haven't actually verified by running the command — "should work" isn't the same as
"works." Commands assume you're at the repo root unless noted.

## Runs locally

- [ ] Fresh virtual environment, dependencies install cleanly:
      `python -m venv .venv ; .\.venv\Scripts\Activate.ps1` then
      `pip install -r requirements.txt`
- [ ] The sample PDFs generate without error:
      `python generate_data.py` (look for 3 PDFs written into `data\`)
- [ ] The CLI indexes, then answers with a citation:
      `python -m pdf_chat ingest data\remote_work_policy.pdf data\solar_system.pdf data\coffee_guide.pdf --index .index`
      then `python -m pdf_chat ask "how many vacation days do I get?" --index .index`
      (expect **25 days**, cited `(remote_work_policy.pdf, p.2)`)
- [ ] The guardrail refuses an off-topic question:
      `python -m pdf_chat ask "what is the capital of France?" --index .index`
      (expect the "I don't know based on the provided documents." refusal)
- [ ] The web app launches and works offline:
      `streamlit run app.py` — upload a PDF from `data\`, ask a question, see an answer
      with a citation and the retrieved chunks in the sidebar. No API key needed.

## Tests pass

- [ ] `pytest` run from the repo root is all green - 25 tests, all offline.
- [ ] You ran it in the fresh venv, not just your everyday one, so you know the deps in
      `requirements.txt` are complete.

## Secrets are clean

- [ ] `.gitignore` contains `.env` and `.index/` (plus `.venv/`, `__pycache__/`).
- [ ] `git status` shows `.env` is NOT tracked.
- [ ] `git ls-files | Select-String ".env"` shows only `.env.example`, never a bare
      `.env`. If a bare `.env` is there, remove it and rotate the key — see the hosting
      guide's troubleshooting section.
- [ ] No API key is hardcoded anywhere in the source.

## README is recruiter-ready

- [ ] Root `README.md` covers: what you built, why it matters, the folder map, the path
      to follow, and copy-pasteable quickstart commands.
- [ ] The CI status badge is at the top.
- [ ] The **live app URL** (from the Space or Streamlit Cloud) is added near the top, so
      a recruiter can click straight through to a working demo.

## Pushed to GitHub with CI

- [ ] Repo created empty on github.com (no auto README/license), named
      `chat-with-your-pdf`, public.
- [ ] `git init` → `git add .` → `git commit` → `git branch -M main` →
      `git remote add origin ...` → `git push -u origin main` all done, from the
      **project root** (the folder containing `hosting/`).
- [ ] `.github/workflows/ci.yml` is committed and pushed.
- [ ] The Actions tab shows a completed run with a green checkmark (it's keyless — all
      tests are offline). If it was red, you read the log and fixed the cause (usually a
      missing dep in `requirements.txt`), then re-ran to green.

## Live app is up

- [ ] Deployed via Path A (Hugging Face Spaces, Streamlit SDK) **or** Path B (Streamlit
      Community Cloud, main file `app.py`).
- [ ] For Path A: `app.py`, the whole `pdf_chat/` folder, and `requirements.txt` are at
      the Space root; the build log is clean.
- [ ] Open the live URL, upload a sample PDF, ask a question — it answers offline with a
      citation. No secret required for the demo to work.
- [ ] (Optional) If you want real-LLM answers online, `GEMINI_API_KEY` is set as a
      Space Secret / Streamlit Secret (NOT in the repo), and `litellm` is added to
      `requirements.txt`.

When every box is ticked, the project is done and presentable. Send the repo link and the
live app link together — the working demo is what gets you the follow-up conversation.
