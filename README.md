
# Pale Blue Dot — AI Universe Video Generator

> A multi-agent system that turns real NASA data into cinematic short films about the universe. Powered by Qwen models on Qwen Cloud, grounded by NASA's public APIs via MCP.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hackathon](https://img.shields.io/badge/Qwen%20Cloud%20Hackathon-Track%202%3A%20AI%20Showrunner-blueviolet)](https://qwencloud-hackathon.devpost.com/)

<!-- ![demo](docs/demo.gif) -->

## What it does

**Pale Blue Dot** is an autonomous AI agent pipeline with a Streamlit chat interface. Type a natural language request — *"reconstruct a windy day on Mars using terrain and weather data"* — and the system builds a short cinematic film grounded entirely in real NASA data.

The Streamlit UI feeds your message to an Orchestrator agent (Qwen on Qwen Cloud) which decides which NASA tools to call, lets you pick 1–3 reference images, writes a short caption per image, generates a visual prompt per scene, and produces one silent video clip per selected frame via Wan / HappyHorse — all streamed back to your browser as it happens.

1. **Chat** — type any astronomy question or video request in natural language
2. **Watch the pipeline run** — each agent step streams status updates to the UI in real time
3. **Get video(s)** — one silent clip per selected NASA image plays inline
4. **Inspect the sources** — NASA data used (images, orbital data, weather, exoplanet parameters) shown alongside

Every frame is anchored to something real. No hallucinated planets, no invented missions — just terrain, weather, orbital context, and event data assembled into a scene.


https://github.com/user-attachments/assets/39d00d64-587d-4d8a-af4d-fba9cfce504c


Music by Kyle Dixon & Michael Stein

## System architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Streamlit Web UI                                │
│  chat input → pipeline status stream → inline video player          │
│  + NASA source panel (images, data used)                            │
└────────────────────────┬────────────────────────────────────────────┘
                         │ user message
┌────────────────────────▼────────────────────────────────────────────┐
│                        Orchestrator Agent                           │
│  (Qwen via Qwen Cloud API + nasa-mcp MCP tools, token-budget loop)  │
└──────┬──────────────────┬──────────────────┬────────────────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌─────────────┐  ┌────────────────┐  ┌─────────────────┐
│  Data Agent │  │ Script Agent   │  │ Storyboard Agent│
│             │  │                │  │                 │
│ nasa-mcp    │  │ scene captions │  │ Visual prompt + │
│ MCP tool    │  │ (1 per image)  │  │ NASA reference  │
│ calls →     │  └────────────────┘  │ frame per scene │
│ raw assets  │                      └────────┬────────┘
└─────────────┘                               │
                                ┌─────────────▼───────────────┐
                                │  Wan 2.7 (i2v / t2v)        │
                                │  one ~10s clip per scene    │
                                └─────────────────────────────┘

All LLM calls  → Qwen Cloud (Alibaba Cloud infrastructure)
MCP server     → Alibaba Cloud hosted instance
Streamlit UI   → runs locally (dev) or Alibaba Cloud ECS (production)
```

## Pipeline Flowchart
flowchart TD
    A[User types request in Streamlit chat] --> B[Phase 1: Data fetch]
    B --> C[Phase 2: Image picker]
    C --> D[Phase 3: Script → Storyboard → Video]
    D --> E[Clip plays inline in browser]

    B --> B1[DataAgent + NASA MCP tools]
    B1 --> B2[output/assets.json]

    C --> C1[User selects NASA reference images]

    D --> D1[ScriptAgent → script.json]
    D1 --> D2[StoryboardAgent → storyboard.json]
    D2 --> D3[VideoGen → output/clips/scene_N.mp4]
    D3 --> D4[episode_manifest.json]


## NASA MCP server — data backbone

The `nasa_mcp/` directory contains a [Model Context Protocol](https://modelcontextprotocol.io) server that exposes NASA's public APIs as typed tool calls. The Orchestrator agent uses these tools to pull real astronomical data into the pipeline.

### Available tools

| Tool | What it provides |
|------|-----------------|
| `get_apod_tool` | Astronomy Picture of the Day for any date back to 1995 |
| `search_apod_tool` | Full-text search across the APOD archive |
| `get_neo_feed_tool` | Near-Earth asteroids and close approach data for a date range |
| `get_neo_lookup_tool` | Detailed orbital data for a specific asteroid |
| `search_image_library_tool` | Search 140,000+ images and videos in the NASA Image Library |
| `get_image_asset_tool` | All size variants and metadata for a specific NASA image asset |
| `search_exoplanets_tool` | Query NASA's Exoplanet Archive (5,000+ confirmed planets) |
| `get_exoplanet_stats_tool` | Summary statistics for the confirmed exoplanet catalog |
| `get_epic_images_tool` | Full-disc Earth photos from DSCOVR's EPIC camera |
| `get_epic_available_dates_tool` | List all dates with EPIC Earth imagery available |
| `get_cme_events_tool` | Coronal mass ejection events from DONKI |
| `get_flr_events_tool` | Solar flare events from DONKI |
| `get_gst_events_tool` | Geomagnetic storm events from DONKI |

### Scene reconstruction sources (planned)

These are the next NASA data sources to wire into the pipeline so a scene can be reconstructed from real terrain, weather, and event context instead of generic prompts.

| Source | What it adds to a reconstructed scene |
|------|---------------------------------------|
| Mars Trek WMTS | Mars terrain, landing-site mosaics, elevation, and camera framing context for believable surface shots |
| EONET | Curated Earth natural events such as storms, fires, dust outbreaks, and cyclones with linked imagery and metadata |
| InSight Mars Weather | Wind, pressure, and temperature measurements from the Martian surface for accurate local conditions |

### MCP server design

- **Feature-first modules** — each NASA API domain lives under `nasa_mcp/features/<feature>/`, with API client, Pydantic input schemas, MCP tool registration, and tests co-located.
- **SQLite cache with TTLs** — immutable data (APOD entries) cached indefinitely; time-sensitive data (NEO feeds) cached 24h. Zero infrastructure required.
- **Pydantic-typed tool schemas** — every tool input is a validated schema surfaced to Qwen for accurate tool selection.
- **stdio transport** — runs as a subprocess of the MCP client; deployable as a long-running Alibaba Cloud Function Compute instance.

## Qwen Cloud integration

| Component | Qwen Cloud service |
|---|---|
| Orchestrator + all agents | Qwen LLM (chat API, tool-calling mode) |
| Video generation | Wan / HappyHorse endpoint on Qwen Cloud |
| MCP server hosting | Alibaba Cloud ECS / Function Compute |
| Asset caching | Alibaba Cloud OSS (production) / SQLite (local dev) |

## Getting started

### Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- [NASA API key](https://api.nasa.gov) (free, instant)
- Qwen Cloud API key (sign up at [qwencloud.com](https://www.qwencloud.com))

### Install

```bash
git clone https://github.com/manvocao0271/nasa-mcp-assisted-ai-video-generation
cd nasa-mcp-assisted-ai-video-generation
uv sync
```

### Configure

```bash
cp .env.example .env
# Set NASA_API_KEY and QWEN_API_KEY in .env
```

### Launch the UI

```bash
uv run streamlit run app.py
```

Open `http://localhost:8501`. Type any request into the chat box, e.g.:

- *"Get me any photo taken on Mars and generate a 10-second video of what it is like on Mars"*
- *"Reconstruct a windy day on Mars using Mars Trek terrain and InSight weather data"*
- *"Show me what happened in space on the day I was born — July 14, 1998"*
- *"Make a short film about the next asteroid passing close to Earth"*
- *"Generate a video about an exoplanet similar to Earth"*
- *"Turn a DONKI solar storm or an EONET wildfire into a dramatic science scene"*

After NASA data is fetched you pick which images to use as visual references. Each subsequent agent step streams a status update to the UI. The generated clip plays inline when complete. NASA images used are shown in the sidebar source panel.

### Run tests

```bash
uv run pytest                          # unit + integration (no network)
uv run pytest -m live                  # live NASA API round-trips (needs NASA_API_KEY)
```

## Project structure

```
app.py                  # Streamlit UI: chat, image picker, pipeline status, inline video
agent/                  # Multi-agent pipeline
  orchestrator.py       # Token-budget-aware pipeline driver
  data_agent.py         # Calls nasa-mcp tools → output/assets.json
  script_agent.py       # Scene captions (1 per selected image) → output/script.json
  storyboard_agent.py   # Visual prompts + NASA ref frames → output/storyboard.json
  video_gen.py          # Wan 2.7 API client (Qwen Cloud) → output/clips/
  qwen_client.py        # OpenAI-compatible Qwen Cloud chat client
nasa_mcp/               # NASA MCP server (data backbone)
  features/
    apod/               # Astronomy Picture of the Day
    donki/              # DONKI space-weather events (CME, flare, geomagnetic storm)
    earth/              # EPIC satellite imagery
    neo/                # Near-Earth Objects
    exoplanets/         # Exoplanet Archive
    image_library/      # NASA Image & Video Library
    mars_trek/          # Mars Trek WMTS (scaffold — not registered yet)
  cache.py              # SQLite TTL cache
  config.py             # Config / env loading
  server.py             # MCP server entry point
output/                 # Generated artifacts (gitignored)
  assets.json           # NASA data fetched for the request
  script.json           # Scene captions (structured JSON, silent video)
  storyboard.json       # Per-scene visual prompts
  clips/                # Generated scene mp4(s)
  episode_manifest.json # Run summary: data sources, token spend, clip paths
tests/                  # Integration tests
```

## Retrieval-Augmented Generation (RAG) & Embeddings Roadmap

Purpose: Ground chat responses in real NASA data and reduce LLM hallucinations by retrieving and citing relevant artifacts (assets.json, script.json, storyboard.json, and NASA tool outputs). Implement in phases: a quick RAG prototype (no external deps), followed by embeddings + vector store for robust, low-latency retrieval.

Phase 1 — Simple RAG (quick)
- Implement a lightweight `Retriever` that reuses `DataAgent` to fetch and summarize NASA tool outputs for a user query.
- Integrate the retriever into `agent/chat_agent.py`: call the retriever before the LLM call, include the top-K summarized passages as a system/context message with citations, then call the LLM. This immediately improves factuality with no new runtime dependencies.
- Add a script to generate short summaries of `output/assets.json` (title, short snippet, source).

Phase 2 — Embeddings + Vector Store (recommended)
- Choose a vector store:
  - Chroma — easy local on-disk index, low ops overhead
  - FAISS — high-performance CPU index for larger corpora
  - Milvus/Weaviate — production-grade managed stores
- Add `agent/embeddings.py` (wraps Qwen embeddings API or `sentence-transformers`) and `agent/vector_store.py` (wraps Chroma/FAISS).
- Create an indexing script `scripts/index_artifacts.py` that:
  1. Reads `output/*.json` (assets, scripts, storyboards) and `nasa_mcp` documentation artifacts,
  2. Extracts passages and metadata,
  3. Computes embeddings,
  4. Persists them into the vector store.
- Query flow: embed the user's query → nearest-neighbor search → retrieve top-K passages with metadata → include passages (summaries + citations) in LLM prompt.

Phase 3 — UI & infra
- Update `app.py` to display retrieved source snippets inline with assistant replies and add a "Cite sources" or "Visualize these sources" action.
- Add background indexing (worker or cron job) to keep the vector store in sync with `output/` runs.
- Add unit/integration tests: `tests/test_retriever.py`, `tests/test_embeddings.py`.

Dependencies (examples)

```bash
# Minimal (Phase 1)
pip install -r requirements.txt

# Phase 2 (embeddings + vector store)
pip install chromadb sentence-transformers
# or for FAISS:
pip install faiss-cpu sentence-transformers
```

Minimal retrieval pseudo-code:

```py
from agent.retriever import Retriever
from agent.qwen_client import QwenClient

retriever = Retriever(vector_store=None)  # Phase 1: uses DataAgent
passages = retriever.retrieve(user_message, top_k=5)

system_msg = """Retrieved sources:
{}
""".format("\n".join([f"[{i+1}] {p['source']}: {p['snippet']}" for i,p in enumerate(passages)]))

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "system", "content": system_msg},
    {"role": "user", "content": user_message},
]

resp = qwen_client.chat(messages=messages, ...)
```

Migration checklist & file additions
- `agent/embeddings.py` — embedding provider wrapper
- `agent/vector_store.py` — Chroma/FAISS wrapper + simple API: `index(docs)`, `query(q, k=5)`
- `agent/retriever.py` — retrieval orchestration (Phase1: DataAgent; Phase2: vector store)
- `scripts/index_artifacts.py` — index existing outputs
- `tests/test_retriever.py` — unit tests

Caveats
- Watch LLM token limits: summarize long tool outputs and include only top-k passages.
- Balance retrieval size and prompt space; prefer short, citation-rich snippets.
- Consider privacy/cost of embeddings API if using Qwen or a managed provider.

## Track 2: AI Showrunner — Hackathon Plan

**Hackathon:** [Global AI Hackathon Series with Qwen Cloud](https://qwencloud-hackathon.devpost.com/) | **Deadline:** Jul 9, 2026 @ 5:00 pm EDT | **Prize:** $7,000 cash + $3,000 cloud credits

This project is competing in **Track 2: AI Showrunner**, which requires an agent that autonomously handles the full short-drama creation pipeline: scriptwriting → storyboarding → video generation → editing. All agents must use **Qwen models via Qwen Cloud**. The backend must be deployed on **Alibaba Cloud**. The NASA MCP server acts as the grounded data backbone; real astronomical events, spacecraft imagery, terrain, weather, and orbital data give every generated film a factual anchor that pure-fiction submissions won't have.

### Concept

**"Pale Blue Dot"** — a series of short (~60-second) space documentaries narrated as dramatic monologues. Each episode is seeded by a real NASA data event (an asteroid close approach, an APOD, a Mars sol's worth of rover photos, a Mars weather report, a solar storm, or a natural event on Earth) and rendered as a cinematic short using Wan/HappyHorse. The Qwen model orchestrates the entire pipeline; NASA MCP tools provide the factual grounding.

### Qwen Cloud integration points

| Component | Qwen Cloud API / service used |
|---|---|
| Orchestrator + all agents | Qwen LLM completions (chat API, tool-calling mode) |
| Storyboard → video | Wan / HappyHorse video generation endpoint on Qwen Cloud |
| Streamlit UI | Alibaba Cloud ECS (production hosting) |
| MCP server hosting | Alibaba Cloud ECS / Function Compute |
| Asset caching | Alibaba Cloud OSS (production) / SQLite (local dev) |

### Agent roles

| Agent | Responsibility | Key outputs |
|---|---|---|
| **Orchestrator** | Drives the pipeline, enforces token budget, writes run manifest | `episode_manifest.json` |
| **Data Agent** | Calls NASA MCP tools, selects the best raw assets for a theme | `assets.json` (URLs, metadata) |
| **Script Agent** | Writes one short visual caption per selected NASA image | `script.json` (scenes, caption, mood, ref_image_url) |
| **Storyboard Agent** | One Wan visual prompt per scene, each locked to its reference frame | `storyboard.json` |
| **Video Gen** | One silent clip per storyboard entry via Wan/HappyHorse on Qwen Cloud | `clips/scene_N.mp4` |

### Data sources → scene reconstruction hooks

| NASA source | Story hook |
|---|---|
| `get_apod_tool` (date-targeted) | Opening image / title card — "On this day in [year], humanity's eye turned to…" |
| `get_neo_feed_tool` | Asteroid close approach as plot inciting event |
| `search_exoplanets_tool` + `get_exoplanet_stats_tool` | "If there is another pale blue dot…" — closing monologue |
| `search_image_library_tool` | B-roll reference frames fed directly to Wan as image-to-video inputs |
| `get_epic_images_tool` | Full-disc Earth shot for opening/closing wide |
| `get_cme_events_tool` / `get_flr_events_tool` / `get_gst_events_tool` | Solar storm beats, aurora-style space-weather visuals, and heliophysics narration |
| Mars Trek WMTS *(planned)* | Mars terrain plate, horizon line, landing-site geography, and topographic camera moves |
| InSight Mars Weather *(planned)* | Wind-driven motion, dustiness, pressure, and temperature for Martian atmosphere scenes |
| EONET *(planned)* | Storm systems, wildfires, dust outbreaks, and other real Earth event backdrops |

### Token budget strategy

Track 2 has the highest token allowance of all tracks, but still enforces a ceiling. Planned mitigations:

1. **Tool-call caching** — `nasa-mcp`'s SQLite cache means repeated asset fetches cost zero tokens after the first call.
2. **Structured outputs** — each agent writes compact JSON/Markdown artifacts; the next agent reads only what it needs rather than the full prior context.
3. **Scene count cap** — default to 3 scenes per episode; the Orchestrator can trim to 2 if budget is running low mid-pipeline.
4. **Prompt compression** — Storyboard Agent sends visual prompts only (≤80 tokens each) to the video gen API, not full narration.

### Judging criteria alignment

| Criterion (weight) | How this project addresses it |
|---|---|
| **Technical Depth & Engineering (30%)** | MCP integration with Qwen Cloud tool-calling; custom NASA MCP server; SQLite cache; Pydantic-typed tool schemas; multi-agent pipeline |
| **Innovation & AI Creativity (30%)** | Real NASA data as factual story anchor; MCP-grounded video generation is a novel combination; modular feature-first architecture |
| **Problem Value & Impact (25%)** | Space education and science communication at zero cost; open-source, deployable by anyone with a NASA API key |
| **Presentation & Documentation (15%)** | Architecture diagram in README; 3-minute demo video; inline tool docs; this plan section |

### Implementation milestones

- [x] **M1 — Qwen Cloud wiring** — `agent/qwen_client.py` with tool-calling against nasa-mcp tools
- [x] **M2 — Data Agent** — MCP tool calls end-to-end; `assets.json` produced for a natural language request
- [x] **M3 — Script Agent** — one caption scene per selected image → `script.json`
- [x] **M4 — Storyboard Agent** — Visual prompts generated; NASA reference frames attached
- [x] **M5 — Video Gen integration** — Wan/HappyHorse API on Qwen Cloud wired; single-scene clip generated
- [x] **M6 — ffmpeg assembly** — removed; N silent clips per run (one per selected image), no concat yet
- [x] **M7 — Orchestrator loop** — Full pipeline runs end-to-end with token-budget guard and resume-from-cache
- [x] **M8 — Streamlit UI** — `app.py` chat interface, NASA image picker, per-step status streaming, inline video player, and NASA source panel
- [ ] **M9 — Alibaba Cloud deployment** — MCP server + Streamlit app deployed to Alibaba Cloud ECS; proof-of-deployment recording created

### Submission checklist

- [ ] Public GitHub repo with MIT license visible in the About section
- [ ] `alibaba_cloud_proof.*` file in repo demonstrating use of Alibaba Cloud services and APIs
- [ ] Architecture diagram (the ASCII diagram above + a rendered PNG version)
- [ ] 3-minute demo video uploaded to YouTube/Vimeo (public) — record using the Streamlit UI: type a request and capture the full pipeline run through to video playback
- [ ] Short recording proving backend runs on Alibaba Cloud (separate from demo)
- [ ] Devpost submission text describing features and functionality
- [ ] Track 2 selected on Devpost submission form
- [ ] Qwen Cloud credits applied (sign up at [qwencloud.com/challenge/hackathon](https://qwencloud.com/challenge/hackathon))

### Known risks

| Risk | Mitigation |
|---|---|
| Wan/HappyHorse rate limits | Queue scenes sequentially; cache completed clips to avoid re-generation |
| Token budget overrun | Structured artifact handoffs + scene count cap (see above) |
| Video gen latency | Async polling with status updates in the Streamlit UI |
| Clip visual consistency | Use the same NASA reference frame as style anchor across all scenes of one episode |
| Alibaba Cloud cold-start latency | Keep MCP server warm with a lightweight health-check ping |

## License

MIT — see [`LICENSE`](LICENSE).
