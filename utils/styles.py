"""Global CSS injection — ChatGPT dark-mode aesthetic for WILL.AI."""
from __future__ import annotations

import streamlit as st

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Global font scale (1.6×) ──────────────────────────────────────────────── */
html {
    font-size: 160% !important;
}

/* ── Base ──────────────────────────────────────────────────────────────────── */
.stApp, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    background-color: #212121 !important;
}

/* ── Remove Streamlit chrome ───────────────────────────────────────────────── */
#MainMenu { visibility: hidden !important; }
header[data-testid="stHeader"] { display: none !important; }
footer { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }

/* ── Sidebar: always visible, fixed 260 px wide ────────────────────────────── */
/* Force the sidebar to stay on-screen even when Streamlit marks it collapsed  */
[data-testid="stSidebar"],
[data-testid="stSidebar"][aria-expanded="false"],
[data-testid="stSidebar"][aria-expanded="true"] {
    transform: translateX(0px) !important;
    min-width: 300px !important;
    max-width: 300px !important;
    background-color: #171717 !important;
    border-right: 1px solid #2a2a2a !important;
    visibility: visible !important;
    display: flex !important;
}
[data-testid="stSidebarContent"] {
    padding: 14px 12px !important;
}
/* Hide resize handle */
[data-testid="stSidebarResizeHandle"] { display: none !important; }
/* Style the collapse toggle instead of hiding it */
[data-testid="stSidebarCollapsedControl"] {
    background-color: #171717 !important;
    border-right: 1px solid #2a2a2a !important;
}
[data-testid="stSidebarCollapsedControl"] svg { color: rgba(255,255,255,0.3) !important; }

/* ── Main content: centered, readable width ────────────────────────────────── */
/* Target multiple Streamlit versions' class names */
[data-testid="stMain"] {
    background-color: #212121 !important;
}
[data-testid="stMainBlockContainer"],
.stMainBlockContainer,
.block-container {
    max-width: 100% !important;
    padding: 2.5rem 4rem 7rem !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
    width: 100% !important;
    box-sizing: border-box !important;
}

/* ── Chat messages ─────────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 0.45rem 0 !important;
    gap: 0.65rem !important;
    border: none !important;
}

/* User: hide avatar, right-align bubble */
[data-testid="stChatMessageAvatarUser"] { display: none !important; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    display: block !important;
    text-align: right !important;
    background: transparent !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
  [data-testid="stChatMessageContent"] {
    display: inline-flex !important;
    text-align: left !important;
    background: #2f2f2f !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 0.7rem 1.1rem !important;
    max-width: 75% !important;
    color: #ececec !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
  [data-testid="stChatMessageContent"] [data-testid="stVerticalBlock"],
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
  [data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] {
    margin: 0 !important; padding: 0 !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
  [data-testid="stChatMessageContent"] p { margin: 0 !important; }

/* Assistant: compact avatar */
[data-testid="stChatMessageAvatarAssistant"] {
    width: 42px !important; min-width: 42px !important; height: 42px !important;
    margin-top: 2px !important;
}

/* Message typography */
[data-testid="stChatMessageContent"] { color: #ececec !important; }
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] {
    font-size: 0.9375rem !important;
    line-height: 1.75 !important;
    color: #ececec !important;
}
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] p {
    margin-bottom: 0.5rem !important;
}
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] p:last-child {
    margin-bottom: 0 !important;
}

/* ── NASA images inside chat: cap height, keep aspect ratio ─────────────────── */
[data-testid="stChatMessage"] [data-testid="stImage"] img {
    max-height: 260px !important;
    width: auto !important;
    max-width: 100% !important;
    object-fit: contain !important;
    display: block !important;
    border-radius: 8px !important;
    background: #111 !important;
}
[data-testid="stChatMessage"] [data-testid="stImage"] {
    overflow: hidden !important;
    border-radius: 8px !important;
}

/* ── Chat input ─────────────────────────────────────────────────────────────── */
/* stBottom spans full main-area width; the areas on each side of the centred  */
/* input block appear as dark bars if this element has its own background.     */
/* Target it three ways to cover all deployed Streamlit versions:              */
/*   1. data-testid (1.29 +)                                                   */
/*   2. class name fallback                                                     */
/*   3. :has() — matches the parent of stBottomBlockContainer regardless of    */
/*      its name, so it works even if the testid changes across versions.       */
[data-testid="stBottom"],
.stBottom,
:has(> [data-testid="stBottomBlockContainer"]) {
    background: #212121 !important;
    background-color: #212121 !important;
    background-image: none !important;
    box-shadow: none !important;
    border-top: none !important;
}
[data-testid="stBottomBlockContainer"] {
    max-width: 100% !important;
    width: 100% !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
    background: #212121 !important;
    border-top: none !important;
    padding: 0.75rem 4rem 1rem !important;
    box-sizing: border-box !important;
}
[data-testid="stChatInput"] [data-baseweb="textarea"] {
    background-color: #2f2f2f !important;
    border-radius: 0 !important;
    border: none !important;
    color: #ececec !important;
    transition: border-color 0.15s !important;
}
[data-testid="stChatInput"] [data-baseweb="textarea"]:focus-within {
    border: none !important;
}

/* ── Sidebar navigation buttons ─────────────────────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
    background: transparent !important;
    border: none !important;
    color: #c8c8c8 !important;
    text-align: center !important;
    justify-content: center !important;
    padding: 0.42rem 0.7rem !important;
    border-radius: 7px !important;
    font-size: 0.85rem !important;
    line-height: 1.45 !important;
    height: auto !important;
    min-height: unset !important;
    width: 100% !important;
    transition: background 0.1s, color 0.1s !important;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
    background: #2a2a2a !important;
    color: #ffffff !important;
}
/* New chat button (primary in sidebar) */
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
    background: transparent !important;
    border: 1px solid #3a3a3a !important;
    color: #ececec !important;
    border-radius: 8px !important;
    justify-content: flex-start !important;
    transition: background 0.1s, border-color 0.1s !important;
    font-size: 0.875rem !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover {
    background: #2a2a2a !important;
    border-color: #555 !important;
}

/* ── Conversation group labels ──────────────────────────────────────────────── */
.conv-group-label {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    text-transform: uppercase !important;
    color: rgba(255,255,255,0.3) !important;
    padding: 0.65rem 0.7rem 0.2rem !important;
    display: block !important;
}

/* ── Suggestion cards (welcome screen buttons) ──────────────────────────────── */
.suggest-card [data-testid="stBaseButton-secondary"] {
    background: #2a2a2a !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 12px !important;
    color: #d1d1d1 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 0.8rem 1rem !important;
    height: auto !important;
    min-height: 58px !important;
    font-size: 0.875rem !important;
    line-height: 1.4 !important;
    transition: background 0.15s, border-color 0.15s !important;
    white-space: normal !important;
}
.suggest-card [data-testid="stBaseButton-secondary"]:hover {
    background: #333 !important;
    border-color: #505050 !important;
    color: #fff !important;
}

/* ── Dividers ───────────────────────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid #2a2a2a !important;
    margin: 0.6rem 0 !important;
}

/* ── Expander (retrieved sources) ───────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #2f2f2f !important;
    border-radius: 8px !important;
    background: transparent !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.8rem !important;
    color: rgba(255,255,255,0.4) !important;
}

/* ── Captions / secondary text ──────────────────────────────────────────────── */
[data-testid="stCaptionContainer"], .stCaption {
    color: #8e8ea0 !important;
    font-size: 0.82rem !important;
}

/* ── Status widget (pipeline) ───────────────────────────────────────────────── */
[data-testid="stStatusWidget"] { color: #ececec !important; }

/* ── Video studio: queue item rows ─────────────────────────────────────────── */
.queue-label {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.65);
    padding: 3px 0 3px 2px;
    line-height: 1.5;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ── Video studio: inline generate button ───────────────────────────────────── */
[data-testid="stMain"] [data-testid="stBaseButton-primary"] {
    border-radius: 10px !important;
}

/* ── Muted footer below chat input ─────────────────────────────────────────── */
.chat-footer {
    text-align: center;
    font-size: 0.75rem;
    color: rgba(255,255,255,0.22);
    margin-top: 0.4rem;
}

/* ── Video Studio side panel ─────────────────────────────────────────────────── */
.studio-title {
    font-size: 0.92rem !important;
    font-weight: 700 !important;
    color: #ececec !important;
    margin: 0 0 2px !important;
    padding: 4px 0 2px !important;
}
.studio-section {
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    color: rgba(255,255,255,0.3) !important;
    margin-bottom: 6px !important;
}
.studio-optional {
    font-weight: 400 !important;
    font-size: 0.63rem !important;
    color: rgba(255,255,255,0.25) !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
}
.studio-empty {
    font-size: 0.8rem !important;
    color: rgba(255,255,255,0.28) !important;
    line-height: 1.55 !important;
    margin: 4px 0 !important;
}
hr.studio-hr {
    margin: 10px 0 !important;
    border-top: 1px solid #2a2a2a !important;
}

/* ── Pipeline step row ───────────────────────────────────────────────────────── */
.pipeline-row {
    display: flex !important;
    align-items: center !important;
    gap: 5px !important;
    margin: 8px 0 6px !important;
}
.pipeline-step {
    flex: 1 !important;
    background: #1c1c1c !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
    padding: 8px 4px !important;
    text-align: center !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    gap: 3px !important;
    min-width: 0 !important;
}
.pipeline-step.step-done {
    border-color: #10a37f !important;
    background: rgba(16,163,127,0.07) !important;
}
.pipeline-step.step-active {
    border-color: #f59e0b !important;
    background: rgba(245,158,11,0.07) !important;
}
.step-icon { font-size: 1rem; line-height: 1; }
.step-name {
    font-size: 0.63rem !important;
    color: rgba(255,255,255,0.45) !important;
    letter-spacing: 0.02em !important;
    white-space: nowrap !important;
}
.step-badge { font-size: 0.68rem !important; }
.badge-done   { color: #10a37f !important; }
.badge-active { color: #f59e0b !important;
    animation: pulse-studio 1.2s ease-in-out infinite !important; }
.badge-pending { color: rgba(255,255,255,0.18) !important; }
.pipeline-arrow {
    color: rgba(255,255,255,0.18) !important;
    font-size: 0.7rem !important;
    flex-shrink: 0 !important;
}
@keyframes pulse-studio {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.25; }
}

/* ── Pipeline execution log ──────────────────────────────────────────────────── */
.pipeline-log {
    background: #1a1a1a;
    border: 1px solid #272727;
    border-radius: 8px;
    padding: 10px 12px;
    margin: 8px 0 4px;
    font-size: 0.76rem;
    line-height: 1.65;
}
.log-stage { }
.log-stage-header {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 2px 0;
}
.log-stage-sep {
    border-top: 1px solid #272727;
    margin-top: 7px;
    padding-top: 7px;
}
.log-icon  { font-size: 0.82rem; flex-shrink: 0; }
.log-label {
    font-weight: 600;
    font-size: 0.76rem;
    color: rgba(255,255,255,0.82);
    letter-spacing: 0.01em;
}
.log-model {
    margin-left: auto;
    font-size: 0.66rem;
    color: rgba(255,255,255,0.3);
    background: rgba(255,255,255,0.05);
    padding: 1px 6px;
    border-radius: 4px;
    font-family: 'SF Mono', 'JetBrains Mono', ui-monospace, monospace;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 62%;
}
.log-status-done    { color: #10a37f; font-size: 0.72rem; flex-shrink: 0; }
.log-status-running { color: #f59e0b; font-size: 0.72rem; flex-shrink: 0;
    animation: pulse-studio 1.2s ease-in-out infinite; }
.log-status-pending { color: rgba(255,255,255,0.18); font-size: 0.72rem; flex-shrink: 0; }
.log-entries { padding-left: 16px; }
.log-entry  { font-size: 0.73rem; color: rgba(255,255,255,0.42); line-height: 1.7; }
.log-done   { color: rgba(255,255,255,0.4); }
.log-running{ color: #f59e0b; }
.log-scene  {
    font-size: 0.70rem;
    color: rgba(255,255,255,0.28);
    font-style: italic;
    padding: 1px 0 3px 8px;
    border-left: 2px solid #2a2a2a;
    margin: 1px 0 4px;
    line-height: 1.5;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.log-warning {
    font-size: 0.73rem;
    color: #f59e0b;
    background: rgba(245,158,11,0.07);
    border-left: 2px solid #f59e0b;
    padding: 5px 9px;
    border-radius: 0 5px 5px 0;
    margin-top: 8px;
    line-height: 1.55;
}

/* ── Clickable queue images ────────────────────────────────────────────────────────────── */
/* Rendered via st.markdown so Streamlit's native image expand is bypassed.   */
.queue-img-cell { border-radius: 8px; overflow: hidden; }
.queue-img-cell img { width: 100%; display: block; border-radius: 8px; }
.queue-img-cell:not(.selected) img { cursor: pointer; }
.queue-img-cell:not(.selected) img:hover {
    outline: 2px solid rgba(255,255,255,0.25) !important;
    outline-offset: -2px !important;
}
.queue-img-caption {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.42);
    text-align: center;
    padding: 4px 0 0;
}
/* Also suppress Streamlit's native image expand button everywhere */
[data-testid="StyledFullScreenButton"] { display: none !important; }
/* Hide the invisible queue-trigger button and all its wrapper divs.
   Targets by the unique help tooltip (title attr) so no fragile DOM-depth
   assumptions are needed — works regardless of Streamlit wrapper changes. */
button[title="queue-trigger"] { display: none !important; pointer-events: none !important; }
[data-testid="stVerticalBlock"] > div:has(button[title="queue-trigger"]) {
    display: none !important;
    height: 0 !important;
    overflow: hidden !important;
    padding: 0 !important;
    margin: 0 !important;
}
"""


def inject_css() -> None:
    """Inject the global ChatGPT-dark stylesheet into the Streamlit app."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)
