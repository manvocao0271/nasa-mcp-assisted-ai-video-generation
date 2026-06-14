# CLAUDE.md — Project Reference

Quick-reference for Claude to avoid re-scanning the repo. Read this first.

## Project in one line

Multi-agent pipeline: user types a natural language astronomy request → Qwen (Qwen Cloud) orchestrates NASA MCP tools → script → storyboard → Wan 2.7 video clip → Streamlit UI.

## Stack

- **Language:** Python 3.12
- **Package manager:** `uv` (`uv sync`, `uv run`)
- **LLM:** Qwen via Qwen Cloud API — `agent/qwen_client.py` (`qwen3.7-plus` data/tools, `qwen-vl-plus-2024-08-13` script/storyboard vision)
- **Video gen:** Wan / HappyHorse on Qwen Cloud — `agent/video_gen.py`
- **MCP:** `mcp[cli]` + `FastMCP`
- **UI:** Streamlit (`app.py`)
- **HTTP:** `httpx` (async in MCP, sync in agents/UI)
- **Validation:** Pydantic v2
- **Cache:** SQLite via `nasa_mcp/cache.py` (TTL-based)
- **Tests:** pytest + pytest-asyncio (~50 unit + integration, no network by default)
- **Build:** hatchling, packages `nasa_mcp/`

## File system

```
CLAUDE.md                   ← this file
README.md                   ← full project docs + hackathon plan
pyproject.toml              ← name: pale-blue-dot, packages: [nasa_mcp]
.env                        ← NASA_API_KEY, QWEN_API_KEY (not committed)
.env.example                ← template for .env
.python-version             ← 3.12

app.py                      ← Streamlit UI: multi-turn chat, astronomer bot, video generator, conversation history

agent/                      ← Multi-agent pipeline & chatbot
  orchestrator.py           ← Drives pipeline, enforces token budget
  chat_agent.py             ← Multi-turn chatbot with conversation context, video suggestions
  run_db.py                 ← SQLite conversation storage (runs, messages, artifacts)
  data_agent.py             ← Calls nasa-mcp tools → output/assets.json
  script_agent.py           ← 3-act narration from assets → output/script.json
  storyboard_agent.py       ← Visual prompts + NASA ref frames → output/storyboard.json
  video_gen.py              ← Wan/HappyHorse Qwen Cloud client → output/clips/scene_N.mp4
  qwen_client.py            ← OpenAI-compatible Qwen Cloud chat client

nasa_mcp/                   ← NASA MCP server (data backbone)
  server.py                 ← FastMCP entry point; registers apod, donki, earth, exoplanets, image_library, neo
  config.py                 ← Config(nasa_api_key, cache_path, request_timeout)
  cache.py                  ← Cache(path); .get(key) .set(key, val, ttl) .stats()
  errors.py                 ← NasaApiError
  features/
    apod/                   ← COMPLETE: get_apod_tool, search_apod_tool
    donki/                  ← COMPLETE: get_cme_events_tool, get_flr_events_tool, get_gst_events_tool
    earth/                  ← COMPLETE: get_epic_images_tool, get_epic_available_dates_tool
    neo/                    ← COMPLETE: get_neo_feed_tool, get_neo_lookup_tool
    exoplanets/             ← COMPLETE: search_exoplanets_tool, get_exoplanet_stats_tool
    image_library/          ← COMPLETE: search_image_library_tool, get_image_asset_tool
    mars_trek/              ← SKELETON (not registered in server.py)

tests/                      ← Integration tests (no network)
  conftest.py               ← Fixtures: tmp_cache_path, test_config, cache, mcp_with_tools
  test_cache.py             ← SQLite cache behavior
  test_server.py            ← Tool registration + description quality
  test_live.py              ← Real NASA API tests (pytest -m live, skipped by default)
```

## Key conventions

- Each feature under `nasa_mcp/features/<name>/` has: `api.py` (httpx calls), `inputs.py` (Pydantic models), `tools.py` (FastMCP registration), `__tests__/`.
- Tools are registered by calling `register_<feature>_tools(mcp, config, cache)` in `server.py`.
- MCP tool names end with `_tool` (e.g. `get_apod_tool`).
- Output artifacts go in `output/` (gitignored): `assets.json`, `script.json`, `storyboard.json`, `clips/scene_N.mp4`, `episode_manifest.json`.
- **One scene per selected NASA image** (1–3 clips per run; 1 text-only scene when no images). Silent video only — no TTS or audio track.

## Pipeline flow (current)

```
CHAT FLOW:
user message (app.py)
  → ChatAgent.answer(message, history)  → Qwen 3.7-plus with conversation context
  → response + video suggestion (if applicable)
  → save to RunDB (for persistence + fine-tuning data)
  ├─ return to chat (default)
  └─ trigger video generation (if user requests)

VIDEO GENERATION FLOW (when triggered from chat):
user clicks "Generate Video"
  → Orchestrator.fetch_data(topic)  → DataAgent  → assets.json  → user picks NASA images
  → Orchestrator.run_pipeline()
      → ScriptAgent      → script.json (N caption scenes)
      → StoryboardAgent  → storyboard.json (N visual prompts)
      → VideoGen         → clips/scene_N.mp4 (one per scene)
      → episode_manifest.json
  → save full run to RunDB (includes video clips)

CONVERSATION PERSISTENCE:
RunDB (output/runs.db)
  ├─ conversations table: (conversation_id, created_at, title)
  └─ runs table: (run_id, user_message, assistant_response, assets, manifest, messages)

## RAG & Embeddings Roadmap (quick reference)

- **Goal:** Ground `ChatAgent` answers in NASA data to reduce hallucinations and provide citations for claims.
- **Phase 1 (quick):** Add `agent/retriever.py` that calls `DataAgent` to fetch and summarize relevant NASA outputs for a query; include top-K summaries in the system prompt before calling the model.
- **Phase 2 (recommended):** Add `agent/embeddings.py` + `agent/vector_store.py` (Chroma/FAISS). Create `scripts/index_artifacts.py` to index `output/*.json` (assets/script/storyboard) into vectors and metadata.
- **Phase 3 (UX & infra):** Show retrieved snippets in the Streamlit UI, add a "Cite sources" toggle, and run a background indexer for new runs.

Files to add/modify (short):
- `agent/embeddings.py` — wrapper for Qwen embeddings or `sentence-transformers`
- `agent/vector_store.py` — Chroma/FAISS wrapper (`index()`, `query()`)
- `agent/retriever.py` — retrieval logic that returns `[{snippet, source, doc_id}]`
- `scripts/index_artifacts.py` — indexing helper to populate vector store from `output/`

Quick commands (example):
```bash
# Phase 2 dependencies
pip install chromadb sentence-transformers
```

Notes:
- Keep retrieved passages short and cite sources to reduce token usage and improve transparency.
- Start Phase 1 to get immediate benefit; Phase 2 adds speed and scale.
```

## Commands

```bash
uv sync                          # install deps
uv run pytest                    # unit + integration (no network)
uv run pytest -m live            # live NASA API tests (needs NASA_API_KEY)
uv run streamlit run app.py      # launch UI
uv run mcp dev nasa_mcp/server.py  # MCP Inspector at localhost:6274
```

## Environment variables

| Variable | Required | Notes |
|---|---|---|
| `NASA_API_KEY` | Yes (prod) | Free at api.nasa.gov. Defaults to `DEMO_KEY` (rate-limited) |
| `QWEN_API_KEY` | Yes | Qwen Cloud / DashScope API key |
| `QWEN_MODEL_DATA` | No | Chat model for Data Agent tool-calling (default `qwen3.7-plus`) |
| `QWEN_MODEL_VISION` | No | VL model for Script/Storyboard (default `qwen-vl-plus-2024-08-13`) |
| `QWEN_CHAT_MODEL` | No | Chat model for ChatAgent (default `qwen3.7-plus`, can set to fine-tuned variant) |
| `NASA_MCP_CACHE_PATH` | No | SQLite cache path (MCP server only) |
| `NASA_MCP_TIMEOUT` | No | NASA HTTP timeout in seconds (default `30`) |

## Registered MCP tools

`get_apod_tool`, `search_apod_tool`, `get_cme_events_tool`, `get_flr_events_tool`, `get_gst_events_tool`, `get_epic_images_tool`, `get_epic_available_dates_tool`, `get_neo_feed_tool`, `get_neo_lookup_tool`, `search_image_library_tool`, `get_image_asset_tool`, `search_exoplanets_tool`, `get_exoplanet_stats_tool`

## What does NOT exist yet

- `mars_trek/` tool implementations (scaffold only)
- EONET and InSight Mars weather integrations
- Multi-clip episode assembly / ffmpeg editing
- Alibaba Cloud production deployment
- `output/` directory (gitignored, created at runtime)

## Chatbot Features (New)

### Multi-Turn Conversation with Persistent History

- **Chat-first UI** — users ask astronomy questions directly; ChatAgent answers with conversation context
- **Persistent storage** — all conversations saved to `output/runs.db` (SQLite)
- **Conversation sidebar** — browse and load past conversations via "History" tab
- **Optional video generation** — ChatAgent detects when user wants visual examples ("show me", "generate", etc.) and suggests video generation

### ChatAgent Architecture

- `ChatAgent.answer(message, conversation_history)` — returns `{answer, should_generate_video, video_topic}`
- System prompt focused on astronomy education + NASA data awareness
- Streams token-by-token via `ChatAgent.chat_with_streaming()` (ready for future UI enhancement)
- Conversation context (last 5 turns) included in system message for coherence

### Fine-Tuning Support

See `FINE_TUNING.md` for:
- System prompt refinement with few-shot examples
- Data collection and JSONL export from RunDB
- Qwen Cloud LoRA fine-tuning workflow
- A/B testing fine-tuned vs baseline models
- Cost and ROI analysis

To fine-tune:
1. Collect 50+ conversation runs via RunDB
2. Export to JSONL: `python scripts/export_training_data.py`
3. Fine-tune via DashScope dashboard or API (~$5–20 for 100 examples)
4. Set `QWEN_CHAT_MODEL=qwen3.7-plus-LoRA-xxxxx` to use fine-tuned variant

### Conversation Phases (Updated)

- **"idle"** — chat mode, accepts user input, calls ChatAgent
- **"video_request"** — user clicked "Generate Video", fetches NASA data
- **"image_selection"** — user selects images for video reference
- **"pipeline"** — video generation running; saves run + artifacts to RunDB
