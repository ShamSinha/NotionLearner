# NotionLearner

Local AI learning inbox: right-click papers/YouTube → Notion, with transcripts, chunked summaries, Feynman notes, and course auto-tagging.

## Features (v0.4)

- **Async jobs** — extension returns immediately; watch progress on `http://localhost:8000`
- **Transcript first** — content is saved to Notion before the LLM finishes
- **YouTube captions**, with optional **Whisper fallback** (`yt-dlp` + mlx-whisper / faster-whisper)
- **Chunked hierarchical summarization** for long lectures
- **Research paper mode** + **PDF text extraction** (PyMuPDF)
- **Feynman study notes** (prose + quotes)
- **Dual models** — `gemma4:e4b` categorize, `qwen3:8b` deep analysis
- **Batch open tabs** into the queue
- **YouTube timestamp deep-links** in Notion
- **Course follow-ups** + local semantic search (Ollama embeddings)
- **Idle Ollama unload** to free RAM on 16GB Macs

## Quick start

```bash
# Models
ollama pull gemma4:e4b
ollama pull qwen3:8b
ollama pull nomic-embed-text   # for local search

# Optional Whisper fallback
brew install yt-dlp
pip install mlx-whisper        # or: pip install faster-whisper

# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env           # fill Notion keys
uvicorn main:app --reload --port 8000
```

Load unpacked extension from `extension/`, set Backend URL + API Secret.

## Right-click actions

| Action | What it does |
|--------|----------------|
| Add to Learning Queue | Fast categorize + Course/Subtopic |
| Add & Summarize | Chunked AI summary |
| Add & Feynman Notes | Deep prose study notes |
| Add as Research Paper | Paper schema (method/loss/results/…) |
| Add ALL open tabs | Batch categorize |

## Notion DB

Use the CSV in `notion/Learning_Queue_import.csv`, connect your integration, set `NOTION_API_KEY` + `NOTION_DATABASE_ID`.

## Dashboard

Open **http://localhost:8000** for jobs, RAM, Ollama load state, and local search.
