
# Pale Blue Dot — AI Universe Video Generator

> A multi-agent system that turns real NASA data into cinematic short films about the universe. Powered by Qwen models on Qwen Cloud, grounded by NASA's public APIs via MCP.

[![Python 3.12+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hackathon](https://img.shields.io/badge/Qwen%20Cloud%20Hackathon-Track%202%3A%20AI%20Showrunner-blueviolet)](https://qwencloud-hackathon.devpost.com/)

<!-- ![demo](docs/demo.gif) -->

## What it does

**Pale Blue Dot** is an autonomous AI agent pipeline with a Streamlit chat interface. Type a natural language request — *"get me any photo taken on Mars and generate a 10-second video of what it is like on Mars"* — and the system builds a short cinematic film grounded entirely in real NASA data.

The Streamlit UI feeds your message to an Orchestrator agent (Qwen on Qwen Cloud) which decides which NASA tools to call, writes a narration script from the results, generates a scene-by-scene storyboard, produces video clips via Wan / HappyHorse, and assembles them into a final film — all streamed back to your browser as it happens.

1. **Chat** — type any astronomy question or video request in natural language
2. **Watch the pipeline run** — each agent step streams status updates to the UI in real time
3. **Get a video** — the finished film plays inline in the browser
4. **Inspect the sources** — NASA data used (images, orbital data, exoplanet parameters) shown alongside

Every frame is anchored to something real. No hallucinated planets, no invented missions.

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
│ nasa-mcp    │  │ 3-act narration│  │ Visual prompt + │
│ MCP tool    │  │ grounded in    │  │ NASA reference  │
│ calls →     │  │ real NASA data │  │ frame attached  │
│ raw assets  │  └────────────────┘  └────────┬────────┘
└─────────────┘                               │
                                ┌─────────────▼───────────────┐
                                │  Wan 2.7 (i2v / t2v)        │
                                │  (Qwen Cloud video gen)     │
                                │  one 10-second clip         │
                                └─────────────────────────────┘

All LLM calls  → Qwen Cloud (Alibaba Cloud infrastructure)
MCP server     → Alibaba Cloud hosted instance
Streamlit UI   → runs locally (dev) or Alibaba Cloud ECS (production)
```

## NASA MCP server — data backbone

The `nasa_mcp/` directory contains a [Model Context Protocol](https://modelcontextprotocol.io) server that exposes NASA's public APIs as typed tool calls. The Orchestrator agent uses these tools to pull real astronomical data into the pipeline.

### Available tools

| Tool | What it provides |
|------|-----------------|
| `get_apod` | Astronomy Picture of the Day for any date back to 1995 |
| `search_apod` | Full-text search across the APOD archive |
| `get_neo_feed` | Near-Earth asteroids and close approach data for a date range |
| `get_neo_lookup` | Detailed orbital data for a specific asteroid |
| `search_image_library` | Search 140,000+ images and videos in the NASA Image Library |
| `get_image_metadata` | Metadata for a specific NASA image asset |
| `search_exoplanets` | Query NASA's Exoplanet Archive (5,000+ confirmed planets) |
| `compare_to_earth` | Compare an exoplanet's parameters to Earth's |
| `get_epic_images` | Full-disc Earth photos from DSCOVR's EPIC camera |
| `get_epic_available_dates` | List all dates with EPIC Earth imagery available |

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

- Python 3.11+
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
- *"Show me what happened in space on the day I was born — July 14, 1998"*
- *"Make a short film about the next asteroid passing close to Earth"*
- *"Generate a video about an exoplanet similar to Earth"*

Each agent step streams a status update to the UI as it runs. The finished video plays inline when complete. The NASA images and data used are shown in a source panel below the video.

### Run tests

```bash
uv run pytest                          # unit + integration (no network)
uv run pytest -m live                  # live NASA API round-trips (needs NASA_API_KEY)
```

## Project structure

```
app.py                  # Streamlit UI entry point
agent/                  # Multi-agent pipeline
  orchestrator.py       # Token-budget-aware pipeline driver
  data_agent.py         # Calls nasa-mcp tools, produces assets.json
  script_agent.py       # Writes 3-act narration from assets.json
  storyboard_agent.py   # Generates visual prompts + attaches reference frames
  video_gen.py          # Wan 2.7 API client (Qwen Cloud) → output/clips/
nasa_mcp/               # NASA MCP server (data backbone)
  features/
    apod/               # Astronomy Picture of the Day
    earth/              # EPIC satellite imagery
    neo/                # Near-Earth Objects
    exoplanets/         # Exoplanet Archive
    image_library/      # NASA Image & Video Library
  cache.py              # SQLite TTL cache
  config.py             # Config / env loading
  server.py             # MCP server entry point
output/                 # Generated artifacts (gitignored)
  assets.json           # NASA data fetched for the request
  script.md             # 3-act narration
  storyboard.json       # Per-scene visual prompts
  clips/                # Individual scene mp4s
  episode_final.mp4     # Assembled film
  episode_manifest.json # Run summary: data sources, token spend, durations
evals/                  # Benchmarks and eval harness
tests/                  # Integration tests
```

## Track 2: AI Showrunner — Hackathon Plan

**Hackathon:** [Global AI Hackathon Series with Qwen Cloud](https://qwencloud-hackathon.devpost.com/) | **Deadline:** Jul 9, 2026 @ 5:00 pm EDT | **Prize:** $7,000 cash + $3,000 cloud credits

This project is competing in **Track 2: AI Showrunner**, which requires an agent that autonomously handles the full short-drama creation pipeline: scriptwriting → storyboarding → video generation → editing. All agents must use **Qwen models via Qwen Cloud**. The backend must be deployed on **Alibaba Cloud**. The NASA MCP server acts as the grounded data backbone; real astronomical events, spacecraft imagery, and orbital data give every generated film a factual anchor that pure-fiction submissions won't have.

### Concept

**"Pale Blue Dot"** — a series of short (~60-second) space documentaries narrated as dramatic monologues. Each episode is seeded by a real NASA data event (an asteroid close approach, an APOD, a Mars sol's worth of rover photos) and rendered as a cinematic short using Wan/HappyHorse. The Qwen model orchestrates the entire pipeline; NASA MCP tools provide the factual grounding.

### Pipeline architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Orchestrator Agent                           │
│  (Qwen via Qwen Cloud API + nasa-mcp MCP tools, token-budget loop)  │
└──────┬──────────────────┬──────────────────┬───────────────────┬────┘
       │                  │                  │                   │
       ▼                  ▼                  ▼                   ▼
┌─────────────┐  ┌────────────────┐  ┌─────────────────┐  ┌───────────┐
│  Data Agent │  │ Script Agent   │  │ Storyboard Agent│  │ Edit Agent│
│             │  │                │  │                 │  │           │
│ nasa-mcp    │  │ Scene / act    │  │ Visual prompt   │  │ Assembly  │
│ MCP tool    │  │ breakdown,     │  │ per scene,      │  │ ordering, │
│ calls →     │  │ narration,     │  │ NASA image as   │  │ music,    │
│ raw assets  │  │ dialogue       │  │ reference frame │  │ captions  │
└─────────────┘  └────────────────┘  └────────┬────────┘  └───────────┘
                                              │
                                ┌─────────────▼───────────────┐
                                │  Video Gen (Wan / HappyHorse│
                                │  via Qwen Cloud)            │
                                │  one clip per scene         │
                                └─────────────────────────────┘

All LLM calls → Qwen Cloud (Alibaba Cloud infrastructure)
MCP server    → Alibaba Cloud hosted instance
```

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
| **Orchestrator** | Drives the pipeline, enforces token budget, stitches results | `episode_manifest.json` |
| **Data Agent** | Calls NASA MCP tools, selects the best raw assets for a theme | `assets.json` (URLs, metadata) |
| **Script Agent** | Writes a 3-act ~300-word narration grounded in the NASA data | `script.md` (scenes, narration, mood) |
| **Storyboard Agent** | Converts each scene into a Wan/HappyHorse-compatible visual prompt, attaches NASA reference frames | `storyboard.json` (prompt, ref_image_url per scene) |
| **Video Gen** | Calls Wan/HappyHorse via Qwen Cloud per scene, polls for completion | `clips/scene_N.mp4` |
| **Edit Agent** | Assembles clips in order, adds captions from script, applies fade transitions | `episode_final.mp4` |

### Data sources → story hooks

| NASA tool | Story hook |
|---|---|
| `get_apod` (date-targeted) | Opening image / title card — "On this day in [year], humanity's eye turned to…" |
| `get_neo_feed` | Asteroid close approach as plot inciting event |
| `search_exoplanets` + `compare_to_earth` | "If there is another pale blue dot…" — closing monologue |
| `search_image_library` | B-roll reference frames fed directly to Wan as `image2video` inputs |
| `get_epic_images` | Full-disc Earth shot for opening/closing wide |

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

- [ ] **M1 — Qwen Cloud wiring** — Qwen Cloud API client working with tool-calling; verified against nasa-mcp tools
- [ ] **M2 — Data Agent** — MCP tool calls work end-to-end; `assets.json` produced for a natural language request
- [ ] **M3 — Script Agent** — 3-act narration generated from `assets.json`; output validated against scene schema
- [ ] **M4 — Storyboard Agent** — Visual prompts generated; NASA reference frames attached
- [ ] **M5 — Video Gen integration** — Wan/HappyHorse API on Qwen Cloud wired; single-scene clip generated
- [ ] **M6 — ffmpeg assembly** ~~(removed — single clip per prompt, no concat needed)~~
- [ ] **M7 — Orchestrator loop** — Full pipeline runs end-to-end with token-budget guard
- [ ] **M8 — Streamlit UI** — `app.py` chat interface with per-step status streaming, inline video player, and NASA source panel
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
| Video gen latency | Async polling; Edit Agent starts assembly as soon as first clip is ready |
| Clip visual consistency | Use the same NASA reference frame as style anchor across all scenes of one episode |
| Alibaba Cloud cold-start latency | Keep MCP server warm with a lightweight health-check ping |

## License

MIT — see [`LICENSE`](LICENSE).
