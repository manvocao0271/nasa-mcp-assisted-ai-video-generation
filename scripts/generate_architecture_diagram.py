"""Generate architecture diagram PNG for WILL.AI hackathon submission."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ── Colours ───────────────────────────────────────────────────────────────────
ORANGE       = "#FF6A00"
BLUE         = "#1A6FBF"
BLUE_LIGHT   = "#4DA6FF"
BLUE_DARK    = "#0D3D6B"
TEAL         = "#00796B"
TEAL_LIGHT   = "#4DB6AC"
PURPLE       = "#6A1B9A"
PURPLE_LIGHT = "#CE93D8"
GRAY_BG      = "#12122A"
GRAY_PANEL   = "#1C1C3A"
GRAY_CARD    = "#0F3460"
RED_DARK     = "#7B2200"
NAVY         = "#001830"
WHITE        = "#FFFFFF"
DIM          = "#BBBBBB"
YELLOW       = "#FFD600"

# ── Canvas  (large so fonts render crisply) ───────────────────────────────────
W, H = 36, 28
fig, ax = plt.subplots(figsize=(W, H))
fig.patch.set_facecolor(GRAY_BG)
ax.set_facecolor(GRAY_BG)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

# ── Drawing helpers ───────────────────────────────────────────────────────────
def box(x, y, w, h, fill, edge=WHITE, lw=1.0, r=0.4, zo=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=zo))

def txt(x, y, s, sz=16, color=WHITE, bold=False, ha="center", va="center", zo=8):
    ax.text(x, y, s, fontsize=sz, color=color, ha=ha, va=va,
            fontweight="bold" if bold else "normal",
            linespacing=1.55, zorder=zo)

def arr(x1, y1, x2, y2, color=DIM, lw=2.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                        connectionstyle="arc3,rad=0.0"), zorder=6)

def section(x, y, w, h, fill, title, tsz=20):
    box(x, y, w, h, fill, lw=1.2, r=0.5, zo=1)
    txt(x + w/2, y + h - 0.55, title, sz=tsz, bold=True, zo=8)

# ══════════════════════════════════════════════════════════════════════════════
# Title
# ══════════════════════════════════════════════════════════════════════════════
txt(W/2, 27.3, "WILL.AI  —  System Architecture", sz=34, bold=True)
txt(W/2, 26.45, "What Infinity Looks Like AI   |   Powered by Alibaba Cloud   |   http://47.85.190.101:8501",
    sz=18, color=ORANGE)

# ══════════════════════════════════════════════════════════════════════════════
# Alibaba Cloud ECS outer container
# ══════════════════════════════════════════════════════════════════════════════
section(0.25, 0.30, 23.8, 25.65, GRAY_PANEL,
        "Alibaba Cloud ECS  —  US Virginia  (ecs.e-c1m1.large  |  2 vCPU  |  2 GB RAM  |  Ubuntu 22.04)", 16)

# ── Streamlit Web UI ──────────────────────────────────────────────────────────
section(0.55, 18.8, 23.2, 6.75, BLUE_DARK, "Streamlit Web UI   (port 8501)", 18)

for i, (title, desc) in enumerate([
    ("Conversations",
     "Browse & resume\npast conversations"),
    ("WILL.AI Chat",
     "Multi-turn astronomy chatbot\nRetriever + live NASA grounding\nImages with [+ Queue] buttons"),
    ("Video Studio",
     "Image queue\nVideo prompt editor\nPipeline status & Cancel\nInline clip playback"),
]):
    px = 0.85 + i * 7.65
    box(px, 19.1, 7.25, 6.05, GRAY_CARD, r=0.35, zo=3)
    txt(px + 3.625, 24.35, title, sz=20, bold=True, color=YELLOW, zo=9)
    txt(px + 3.625, 22.55, desc,  sz=16, color=DIM,    zo=9)

# ── Agent Pipeline ────────────────────────────────────────────────────────────
section(0.55, 10.5, 23.2, 7.9, "#0D2B45", "Agent Pipeline", 18)

for i, (name, desc) in enumerate([
    ("ChatAgent",       "Streams answer\ntokens"),
    ("Retriever",       "Top-K NASA\npassages"),
    ("DataAgent",       "Calls MCP\ntools"),
    ("ScriptAgent",     "Scene\ncaptions"),
    ("StoryboardAgent", "Visual\nprompts"),
    ("VideoGen",        "Wan 2.7\nclips"),
]):
    bx = 0.85 + i * 3.88
    box(bx, 10.8, 3.65, 7.25, BLUE, r=0.3, zo=3)
    txt(bx + 1.825, 14.85, name, sz=17, bold=True, color=YELLOW, zo=9)
    txt(bx + 1.825, 13.3,  desc, sz=15, color=DIM,    zo=9)
    if i < 5:
        arr(bx + 3.65, 14.4, bx + 3.88, 14.4, color=ORANGE, lw=2.8)

# ── Local Storage ─────────────────────────────────────────────────────────────
section(0.55, 1.0, 11.0, 9.1, "#1B2631", "Local Storage", 18)

for sx, title, desc in [
    (0.85, "SQLite RunDB",
     "Conversations\nMessages\nPipeline runs\nArtifacts"),
    (6.25, "SQLite Cache",
     "NASA API responses\nTTL-based caching\n24 h  to  indefinite"),
]:
    box(sx, 1.3, 4.9, 8.35, TEAL, r=0.3, zo=3)
    txt(sx + 2.45, 8.65, title, sz=18, bold=True, color=WHITE, zo=9)
    txt(sx + 2.45, 6.55, desc,  sz=15, color=DIM,   zo=9)

# ── NASA MCP Server ───────────────────────────────────────────────────────────
section(11.9, 1.0, 11.85, 9.1, "#1B2631",
        "NASA MCP Server  (FastMCP)", 18)

TOOLS = [
    "get_apod_tool",             "search_apod_tool",
    "get_neo_feed_tool",         "get_neo_lookup_tool",
    "search_image_library_tool", "get_image_asset_tool",
    "search_exoplanets_tool",    "get_exoplanet_stats_tool",
    "get_epic_images_tool",      "get_cme_events_tool",
    "get_flr_events_tool",       "get_gst_events_tool",
    "get_epic_available_dates_tool",
]
for i, t in enumerate(TOOLS):
    col = i % 2
    row = i // 2
    tx = 12.15 + col * 5.85
    ty = 8.55 - row * 0.88
    box(tx, ty - 0.3, 5.5, 0.72, PURPLE, lw=0.5, r=0.18, zo=3)
    txt(tx + 2.75, ty + 0.07, t, sz=13, color=WHITE, zo=9)

# ══════════════════════════════════════════════════════════════════════════════
# Alibaba Cloud DashScope (external)
# ══════════════════════════════════════════════════════════════════════════════
section(24.5, 13.5, 11.2, 12.15, "#1C0A00",
        "Alibaba Cloud\nDashScope API", 20)
txt(30.1, 25.05, "dashscope-intl.aliyuncs.com", sz=15, color=ORANGE)

for by, title, sub1, sub2 in [
    (20.7, "Qwen 3.7-plus",        "LLM inference & tool-calling",  "/compatible-mode/v1"),
    (17.25,"Qwen VL-plus (Vision)", "Script & Storyboard Agents",    "/compatible-mode/v1"),
    (14.2, "Wan 2.7  i2v / t2v",   "Video generation (async poll)", "/api/v1/tasks"),
]:
    box(24.8, by, 10.6, 2.9, RED_DARK, r=0.3, zo=3)
    txt(30.1, by + 2.3,  title, sz=18, bold=True, color=YELLOW, zo=9)
    txt(30.1, by + 1.5,  sub1,  sz=15, color=DIM,    zo=9)
    txt(30.1, by + 0.7,  sub2,  sz=14, color=ORANGE, zo=9)

# ══════════════════════════════════════════════════════════════════════════════
# NASA Public APIs (external)
# ══════════════════════════════════════════════════════════════════════════════
section(24.5, 0.30, 11.2, 12.8, NAVY,
        "NASA Public APIs\n(api.nasa.gov)", 20)

for i, (name, sub) in enumerate([
    ("APOD Archive",       "1995 to present"),
    ("Near-Earth Objects", "orbital & approach data"),
    ("NASA Image Library", "140,000+ assets"),
    ("DSCOVR EPIC",        "full-disc Earth imagery"),
    ("Exoplanet Archive",  "5,000+ confirmed planets"),
    ("DONKI Space Weather","CME  /  Flares  /  GST"),
]):
    col = i % 2
    row = i // 2
    nx = 24.8 + col * 5.55
    ny = 10.3 - row * 2.8
    box(nx, ny, 5.25, 2.25, BLUE_DARK, r=0.25, zo=3)
    txt(nx + 2.625, ny + 1.65, name, sz=15, bold=True, color=WHITE, zo=9)
    txt(nx + 2.625, ny + 0.75, sub,  sz=13, color=DIM,   zo=9)

# ══════════════════════════════════════════════════════════════════════════════
# Arrows
# ══════════════════════════════════════════════════════════════════════════════
arr(11.9,  18.8,  11.9,  18.05, color=BLUE_LIGHT, lw=3.0)   # UI -> Agents
arr(11.9,  18.05, 11.9,  18.8,  color=BLUE_LIGHT, lw=2.0)   # Agents -> UI
arr(2.7,   10.8,  3.5,   9.45,  color=TEAL_LIGHT,  lw=2.5)  # ChatAgent -> RunDB
arr(11.0,  12.5,  11.9,  11.8,  color=PURPLE_LIGHT,lw=2.5)  # DataAgent -> MCP
arr(24.0,  15.0,  24.5,  18.2,  color=ORANGE, lw=3.2)        # Agents -> Qwen
arr(24.0,  13.5,  24.5,  15.5,  color=ORANGE, lw=3.2)        # VideoGen -> Wan
arr(18.65,  1.0,  24.5,   6.5,  color=BLUE_LIGHT,  lw=2.5)  # MCP -> NASA APIs

# ══════════════════════════════════════════════════════════════════════════════
# Legend
# ══════════════════════════════════════════════════════════════════════════════
for i, (c, label) in enumerate([
    (ORANGE,       "Alibaba Cloud API call"),
    (BLUE_LIGHT,   "UI / data flow"),
    (TEAL_LIGHT,   "Database read/write"),
    (PURPLE_LIGHT, "MCP tool call"),
]):
    lx = 0.6 + i * 5.85
    box(lx, 0.35, 0.7, 0.5, c, lw=0.4, r=0.1, zo=7)
    txt(lx + 1.05, 0.60, label, sz=15, color=DIM, ha="left", zo=9)

# ══════════════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════════════
out = Path("docs/architecture.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=GRAY_BG, pad_inches=0.2)
plt.close()
print(f"Saved -> {out.resolve()}")
