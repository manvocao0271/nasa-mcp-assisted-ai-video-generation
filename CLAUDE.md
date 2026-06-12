# CLAUDE.md — Project Reference

Quick-reference for Claude to avoid re-scanning the repo. Read this first.

## Project in one line

Multi-agent pipeline: user types a natural language astronomy request → Qwen (Qwen Cloud) orchestrates NASA MCP tools → script → storyboard → Wan 2.7 video clip → Streamlit UI.

## Stack

- **Language:** Python 3.12
- **Package manager:** `uv` (`uv sync`, `uv run`)
- **LLM:** Qwen via Qwen Cloud API (tool-calling mode) — `agent/qwen_client.py`
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

app.py                      ← Streamlit UI: chat, image picker, pipeline status, inline video

agent/                      ← Multi-agent pipeline
  orchestrator.py           ← Drives pipeline, enforces token budget
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
- Current pipeline generates **one** video clip per run (`VideoGen.MAX_CLIPS = 1`). Multi-scene assembly is not implemented yet.

## Pipeline flow (current)

```
user message (app.py)
  → Orchestrator.fetch_data()  → DataAgent  → assets.json  → user picks NASA images
  → Orchestrator.run_pipeline()
      → ScriptAgent      → script.json
      → StoryboardAgent  → storyboard.json
      → VideoGen         → clips/scene_0.mp4
      → episode_manifest.json
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
