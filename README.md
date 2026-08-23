# NotionLearner

Local AI learning inbox: right-click papers/YouTube → Notion, with transcripts, chunked summaries, Feynman notes, and course auto-tagging.

## Features (v0.5)

- **Async jobs** — extension returns immediately; watch progress on `http://localhost:8000`
- **Transcript first** — content is saved to Notion before the LLM finishes
- **YouTube captions**, with optional **Whisper fallback** (`yt-dlp` + mlx-whisper / faster-whisper)
- **Chunked hierarchical summarization** for long lectures
- **Research paper mode** + **PDF text extraction** (PyMuPDF)
- **Grounded learning notes** with evidence, mental models, misconceptions, and active recall
- **Validated structured output** with automatic schema-guided retry
- **Feynman explanations** grounded in evidence sampled across the source
- **Dual models** — `qwen3:4b` fast categorize, `qwen3:8b` deep analysis
- **Duplicate-aware reprocessing** — the same URL updates its existing Notion page
- **Batch open tabs** into the queue
- **YouTube timestamp deep-links** in Notion
- **Course follow-ups** + local semantic search (Ollama embeddings)
- **Idle Ollama unload** to free RAM on 16GB Macs

## Quick start

```bash
# Models
ollama pull qwen3:4b
ollama pull qwen3:8b
ollama pull nomic-embed-text   # for local search

# Optional Whisper fallback
brew install yt-dlp
pip install mlx-whisper        # or: pip install faster-whisper

# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload --port 8000
```

### Configure `.env`

Create your local environment file from the included template:

```bash
cd backend
cp .env.example .env
```

Open `backend/.env` and set these required values:

```dotenv
NOTION_API_KEY=secret_your_notion_integration_key
NOTION_DATABASE_ID=your_notion_database_id
API_SECRET=choose-a-private-local-secret
```

Connect the Notion integration to your Learning Queue database before starting the
backend. Keep `.env` private—it is ignored by Git and must not be committed.

## Set up the Chrome extension

1. Make sure Ollama and the NotionLearner backend are running:

   ```bash
   ollama serve

   cd /path/to/NotionLearner/backend
   source .venv/bin/activate
   uvicorn main:app --reload --port 8000
   ```

2. Open `chrome://extensions` in Chrome.
3. Enable **Developer mode** in the upper-right corner.
4. Click **Load unpacked** and select the project's `NotionLearner/extension` folder.
5. Optionally pin NotionLearner from Chrome's Extensions menu for quicker access.
6. Click the NotionLearner toolbar icon and expand **Settings and models**.
7. Set **Backend URL** to `http://localhost:8000`.
8. Set **API Secret** to the exact same value used for `API_SECRET` in
   `backend/.env`.
9. Confirm the defaults are `qwen3:4b` for **Save / categorize** and `qwen3:8b`
   for **Analyze / explain**, then click **Save settings**.

The popup now provides four actions for the current page:

- **Save** — extract the source and quickly categorize it with Qwen 4B.
- **Summarize** — generate grounded study notes with Qwen 8B.
- **Explain** — create a step-by-step Feynman explanation with Qwen 8B.
- **Research paper** — extract the method, experiments, results, and limitations.

You can also right-click a page, link, or selected text to use the same actions.
Chrome requests access to page content because the extension sends the selected page's
HTML to your local backend for extraction; it is not sent to a hosted AI service.

After changing files inside `extension/`, return to `chrome://extensions` and click the
**Reload** button on the NotionLearner card. If the popup says the backend is unreachable,
check `http://localhost:8000/health` and confirm the Backend URL and API Secret match.

## Right-click actions

| Action | What it does |
|--------|----------------|
| Add to Learning Queue | Fast categorize + Course/Subtopic |
| Add & Summarize | Chunked AI summary |
| Add & Explain (Feynman) | Grounded step-by-step teaching note |
| Add as Research Paper | Paper schema (method/loss/results/…) |
| Add ALL open tabs | Batch categorize |

## Notion DB

Use the CSV in `notion/Learning_Queue_import.csv`, connect your integration, set `NOTION_API_KEY` + `NOTION_DATABASE_ID`.

## Dashboard

Open **http://localhost:8000** for jobs, RAM, Ollama load state, and local search.
