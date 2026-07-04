"""Global CSS injection — ChatGPT dark-mode aesthetic for WILL.AI."""
from __future__ import annotations

import streamlit as st

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

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
    min-width: 258px !important;
    max-width: 258px !important;
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
    max-width: 800px !important;
    padding: 1.5rem 2rem 7rem !important;
    margin-left: auto !important;
    margin-right: auto !important;
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
    width: 28px !important; min-width: 28px !important; height: 28px !important;
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
/* stBottomBlockContainer is the correct data-testid in Streamlit 1.58        */
[data-testid="stBottom"] {
    background-color: #212121 !important;
}
[data-testid="stBottomBlockContainer"] {
    max-width: 800px !important;
    width: 100% !important;
    margin-left: auto !important;
    margin-right: auto !important;
    background-color: #212121 !important;
    border-top: none !important;
    padding: 0.75rem 0 1rem !important;
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
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 0.42rem 0.7rem !important;
    border-radius: 7px !important;
    font-size: 0.85rem !important;
    line-height: 1.45 !important;
    height: auto !important;
    min-height: unset !important;
    width: 100% !important;
    transition: background 0.1s, color 0.1s !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
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
.queue-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 4px;
    border-bottom: 1px solid #2a2a2a;
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
"""


def inject_css() -> None:
    """Inject the global ChatGPT-dark stylesheet into the Streamlit app."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)
