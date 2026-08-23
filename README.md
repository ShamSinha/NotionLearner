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

Complete the Notion connection and database setup below before starting the backend.
Keep `.env` private—it is ignored by Git and must not be committed.

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

## Set up the Notion connection and database

### 1. Create an internal Notion connection

1. Open Notion's [Internal connections guide](https://developers.notion.com/guides/get-started/internal-connections)
   and navigate to the Developer portal.
2. Under **Build**, select **Internal connections**, then click
   **Create a new connection**.
3. Name it `NotionLearner` and select the workspace that will contain the Learning
   Queue. Notion requires a Workspace Owner to create an internal connection.
4. In the connection's **Configuration** tab, enable these content capabilities:

   - **Read content** — required to find duplicates and inspect the database schema.
   - **Insert content** — required to create new learning pages and blocks.
   - **Update content** — required to write AI analysis and reprocess existing pages.

5. Copy the **Installation access token** and place it in `backend/.env`:

   ```dotenv
   NOTION_API_KEY=your_installation_access_token
   ```

Do not paste this token into the extension or commit it to Git. The extension's
`API_SECRET` is a separate local password and is not your Notion token.

### 2. Create the Learning Queue database

The easiest option is to import the included template:

1. In Notion desktop or web, open **Settings → Import → CSV**. You can also type
   `/csv` on a page.
2. Upload `notion/Learning_Queue_import.csv` from this repository.
3. Choose where the new database should be created and name it `Learning Queue`.
4. Open the resulting database as a full page.

CSV import creates rows as pages and columns as properties. After importing, confirm
these property names and types; the names are case-sensitive:

| Property | Notion type |
|---|---|
| `Name` | Title |
| `URL` | URL |
| `Type` | Select |
| `Status` | Select |
| `Course` | Select |
| `Subtopic` | Text |
| `Domain` | Select |
| `Priority` | Select |
| `Estimated Time` | Number |
| `AI Summary` | Text |
| `Key Concepts` | Text |
| `Prerequisites` | Text |
| `Questions` | Text |
| `ChatGPT Link` | URL |
| `Personal Notes` | Text |
| `Completed On` | Date |

`Transcript` is optional. If you add it, make it a **Text** property; the complete
transcript is always written into the Notion page body even when this property is absent.

If you already have a Learning Queue database, you can recreate the properties above
instead of importing the sample CSV. Notion's **Merge with CSV** operation adds rows; it
does not update matching existing rows, so avoid importing the sample twice.

### 3. Grant the connection access

A new internal connection has no access to any page by default.

1. Open the full-page `Learning Queue` database.
2. Click the `•••` menu in the top-right corner.
3. Select **Connections → Add connection**.
4. Search for `NotionLearner`, select it, and confirm access.

Alternatively, open the connection's **Content access** tab in the Developer portal,
click **Edit access**, and select the Learning Queue database. Without this step, the
Notion API normally returns an `object_not_found` or permission error.

### 4. Find the Notion database ID

1. Open `Learning Queue` as a full-page database—not merely a linked view.
2. Click **Share → Copy link**.
3. Paste the URL into a text editor. It will look similar to:

   ```text
   https://www.notion.so/my-workspace/248104cd477e80fdb757e945d38000bd?v=148104cd477e80bb928f000ce197ddf2
   ```

4. The database ID is the 32-character value immediately before `?v=`:

   ```text
   248104cd477e80fdb757e945d38000bd
   ```

   Notion may also display it as a 36-character UUID containing hyphens. Either form is
   accepted. Do **not** use the value after `?v=`; that is the database view ID.

5. Add the database ID to `backend/.env`:

   ```dotenv
   NOTION_DATABASE_ID=248104cd477e80fdb757e945d38000bd
   ```

NotionLearner resolves the database's first data source automatically, so configure the
Learning Queue as the database's first or only data source.

### 5. Verify the Notion connection

From the repository root, run this read-only check:

```bash
cd backend
source .venv/bin/activate
python -c "from services.notion_client import _resolve_data_source; _, title, props = _resolve_data_source(); print('Connected:', title); print('Properties:', sorted(props))"
```

A successful result prints `Connected: Name` followed by the database properties. If it
fails, check the integration token, database ID, connection access, and property types.

Official references: [internal connections](https://developers.notion.com/guides/get-started/internal-connections),
[working with databases and IDs](https://developers.notion.com/guides/data-apis/working-with-databases),
and [CSV import](https://www.notion.com/help/import-data-into-notion).

## Dashboard

Open **http://localhost:8000** for jobs, RAM, Ollama load state, and local search.
