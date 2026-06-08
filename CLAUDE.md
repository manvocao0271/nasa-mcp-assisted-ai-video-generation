# CLAUDE.md — Project Reference

Quick-reference for Claude to avoid re-scanning the repo. Read this first.

## Project in one line

Multi-agent pipeline: user types a natural language astronomy request → Qwen (Qwen Cloud) orchestrates NASA MCP tools → script → storyboard → Wan/HappyHorse video clips → ffmpeg assembled film → Streamlit UI.

## Stack

- **Language:** Python 3.12
- **Package manager:** `uv` (`uv sync`, `uv run`)
- **LLM:** Qwen via Qwen Cloud API (tool-calling mode)
- **Video gen:** Wan / HappyHorse on Qwen Cloud
- **MCP:** `mcp[cli]` + `FastMCP`
- **UI:** Streamlit (`app.py`) — not yet created
- **HTTP:** `httpx` (async)
- **Validation:** Pydantic v2
- **Cache:** SQLite via `nasa_mcp/cache.py` (TTL-based)
- **Tests:** pytest + pytest-asyncio
- **Build:** hatchling, packages `nasa_mcp/`

## File system

```
CLAUDE.md                   ← this file
README.md                   ← full project docs + hackathon plan
pyproject.toml              ← name: pale-blue-dot, packages: [nasa_mcp]
.env                        ← NASA_API_KEY, QWEN_API_KEY (not committed)
.python-version             ← 3.12

app.py                      ← Streamlit UI entry point (TO CREATE)

agent/                      ← Multi-agent pipeline (TO CREATE)
  orchestrator.py           ← Drives pipeline, enforces token budget
  data_agent.py             ← Calls nasa-mcp tools → assets.json
  script_agent.py           ← 3-act narration from assets.json → script.md
  storyboard_agent.py       ← Visual prompts + NASA ref frames → storyboard.json
  video_gen.py              ← Wan/HappyHorse Qwen Cloud client → clips/
  edit_agent.py             ← ffmpeg: concat + captions + fades → episode_final.mp4

nasa_mcp/                   ← NASA MCP server (data backbone, COMPLETE)
  server.py                 ← FastMCP entry point; registers apod, earth, neo tools
  config.py                 ← Config(nasa_api_key, cache_path, request_timeout)
  cache.py                  ← Cache(path); .get(key) .set(key, val, ttl) .stats()
  errors.py                 ← NasaApiError
  features/
    apod/                   ← COMPLETE: get_apod, search_apod
      api.py, inputs.py, tools.py, __tests__/test_apod.py
    earth/                  ← COMPLETE: get_epic_images, get_epic_available_dates
      api.py, inputs.py, tools.py, __tests__/test_earth.py
    neo/                    ← COMPLETE: get_neo_feed, get_neo_lookup
      api.py, inputs.py, tools.py, __tests__/test_neo.py
    mars_rovers/            ← BROKEN (upstream NASA proxy archived)
      api.py, inputs.py, tools.py, __tests__/test_mars_rovers.py
      ⚠ NOT registered in server.py. Fallback: search_image_library
    exoplanets/             ← STUB (__init__.py only)
    image_library/          ← STUB (__init__.py only)

tests/                      ← Integration tests (no network)
  conftest.py               ← Fixtures: tmp_cache_path, test_config, cache, mcp_with_tools
  test_cache.py             ← SQLite cache behavior
  test_server.py            ← Tool registration + description quality
  test_live.py              ← Real NASA API tests (pytest -m live, skipped by default)
```

## Key conventions

- Each feature under `nasa_mcp/features/<name>/` has: `api.py` (httpx calls), `inputs.py` (Pydantic models), `tools.py` (FastMCP registration), `__tests__/`.
- Tools are registered by calling `register_<feature>_tools(mcp, config, cache)` in `server.py`.
- Mars rovers are intentionally NOT registered in `server.py` — keep it that way until upstream is fixed.
- `exoplanets/` and `image_library/` are stub directories for planned features.
- `agent/` and `app.py` do not exist yet — next major work.
- Output artifacts go in `output/` (gitignored): `assets.json`, `script.md`, `storyboard.json`, `clips/`, `episode_final.mp4`, `episode_manifest.json`.

## Commands

```bash
uv sync                          # install deps
uv run pytest                    # unit + integration (33 tests, no network)
uv run pytest -m live            # live NASA API tests (needs NASA_API_KEY)
uv run streamlit run app.py      # launch UI (once app.py exists)
uv run mcp dev nasa_mcp/server.py  # MCP Inspector at localhost:6274
```

## Environment variables

| Variable | Required | Notes |
|---|---|---|
| `NASA_API_KEY` | Yes (prod) | Free at api.nasa.gov. Defaults to `DEMO_KEY` (rate-limited) |
| `QWEN_API_KEY` | Yes | Qwen Cloud API key |

## Registered MCP tools (as of now)

`get_apod_tool`, `search_apod_tool`, `get_epic_images_tool`, `get_epic_available_dates_tool`, `get_neo_feed_tool`, `get_neo_lookup_tool`

## What does NOT exist yet

- `app.py` — Streamlit chatbot UI
- `agent/` — entire multi-agent pipeline
- `.env.example` — should be created alongside `agent/`
- Qwen Cloud API client
- Wan/HappyHorse video gen client
- `output/` directory (gitignored, created at runtime)
