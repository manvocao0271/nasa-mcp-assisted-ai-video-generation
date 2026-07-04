"""Chat page — WILL.AI astronomy chat with live NASA data grounding."""
from __future__ import annotations

import os
import uuid
from datetime import date, datetime, timedelta

import streamlit as st
from openai import AuthenticationError, APIError

from agent.chat_agent import ChatAgent
from agent.qwen_client import QwenClient, MODEL_CHAT
from agent.run_db import Message
from utils.helpers import fetch_thumb

# ── Environment ────────────────────────────────────────────────────────────────
NASA_API_KEY = os.environ.get("NASA_API_KEY", "DEMO_KEY")
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")

# Session state is already initialised by app.py
_db = st.session_state.run_db

# ── Suggested prompts ──────────────────────────────────────────────────────────
_SUGGESTIONS = [
    "What's today's Astronomy Picture of the Day?",
    "Show me the latest solar flare activity",
    "Find exoplanets in the habitable zone",
    "What asteroids are passing near Earth this week?",
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def _queue_image(img: dict, description: str, data_context: dict) -> None:
    """Add image to video queue, deduplicating by URL."""
    url = img.get("url", "")
    if not url:
        return
    if any(item["url"] == url for item in st.session_state.video_queue):
        return
    st.session_state.video_queue.append({
        "url": url,
        "thumb_url": img.get("thumb_url", ""),
        "caption": img.get("caption", ""),
        "source": img.get("source", "NASA"),
        "data_context": data_context,
        "description": description,
    })


def _save_chat_turn(user_message: str, assistant_response: str) -> None:
    """Persist a chat turn to RunDB."""
    conv_id = st.session_state.conversation_id
    history_before = _db.get_conversation_history(conv_id)
    is_first = len(history_before) == 0
    _db.save_run(
        run_id=str(uuid.uuid4()),
        conversation_id=conv_id,
        user_message=user_message,
        assistant_response=assistant_response,
        assets={},
        manifest={},
        messages=[
            Message(role=m["role"], content=m["content"], timestamp=datetime.now().isoformat())
            for m in st.session_state.messages
        ],
    )
    if is_first:
        _db.set_conversation_title(conv_id, user_message[:120])


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # WILL.AI branding
    st.markdown(
        """
        <div style="padding: 6px 4px 16px; display:flex; align-items:center; gap:10px;">
            <div style="font-size:1.15rem; font-weight:700; color:#ececec; letter-spacing:-0.02em;">
                WILL.AI
            </div>
            <div style="font-size:0.68rem; color:rgba(255,255,255,0.3); margin-top:1px;">
                What Infinity Looks Like
            </div>
        </div>
        <div style="
            display:inline-flex; align-items:center; gap:5px;
            background:linear-gradient(90deg,#FF6A00,#EE0979);
            color:#fff; font-size:10px; font-weight:700; letter-spacing:0.04em;
            padding:3px 8px; border-radius:5px; margin-bottom:12px;">
            &#9729; Powered by Alibaba Cloud
        </div>
        <div style="font-size:10px; color:rgba(255,255,255,0.28); margin-bottom:14px; line-height:1.6;">
            LLM &amp; Video: DashScope API<br>
            Models: Qwen 3.7-plus · Wan 2.7 i2v/t2v
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("＋  New chat", use_container_width=True, key="new_chat_btn", type="primary"):
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.markdown('<hr>', unsafe_allow_html=True)

    # ── Conversation list grouped by date ──────────────────────────────────
    conversations = _db.list_conversations()
    if conversations:
        today = date.today()
        yesterday = today - timedelta(days=1)
        week_ago = today - timedelta(days=7)

        groups: dict[str, list] = {
            "Today": [],
            "Yesterday": [],
            "Previous 7 Days": [],
            "Earlier": [],
        }
        for conv in conversations:
            try:
                conv_date = datetime.fromisoformat(conv["created_at"]).date()
            except Exception:
                groups["Earlier"].append(conv)
                continue
            if conv_date == today:
                groups["Today"].append(conv)
            elif conv_date == yesterday:
                groups["Yesterday"].append(conv)
            elif conv_date >= week_ago:
                groups["Previous 7 Days"].append(conv)
            else:
                groups["Earlier"].append(conv)

        for group_name, convs in groups.items():
            if not convs:
                continue
            st.markdown(
                f'<span class="conv-group-label">{group_name}</span>',
                unsafe_allow_html=True,
            )
            for conv in convs:
                title = (conv.get("title") or "Untitled")[:36]
                if st.button(title, use_container_width=True, key=f"conv_{conv['conversation_id']}"):
                    st.session_state.conversation_id = conv["conversation_id"]
                    history = _db.get_conversation_history(conv["conversation_id"])
                    st.session_state.messages = []
                    for run in history:
                        st.session_state.messages.append(
                            {"role": "user", "content": run["user_message"]}
                        )
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": run["assistant_response"],
                            "assets": run.get("assets") or {},
                            "retrieved_passages": [],
                        })
                    st.rerun()
    else:
        st.caption("No conversations yet.")

    # ── Video Studio link ──────────────────────────────────────────────────
    st.markdown('<hr style="margin-top:auto;">', unsafe_allow_html=True)
    _q_count = len(st.session_state.video_queue)
    _studio_lbl = f"🎬  Video Studio" + (f"  ({_q_count})" if _q_count else "")
    if st.button(_studio_lbl, use_container_width=True, key="goto_studio"):
        st.switch_page("pages/video_studio.py")

# ── Main chat area ─────────────────────────────────────────────────────────────

user_input = st.chat_input("Ask anything about the universe…")

# Fire a suggestion-click prompt if one is pending
if not user_input and st.session_state.get("pending_prompt"):
    user_input = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if not st.session_state.messages and not user_input:
    # ── Welcome screen ─────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="height:42vh; display:flex; flex-direction:column;
                    align-items:center; justify-content:flex-end;
                    text-align:center; gap:0.9rem; padding-bottom:1.5rem;">
            <p style="font-size:2rem; font-weight:700; color:#ececec;
                      margin:0; letter-spacing:-0.03em; line-height:1.1;">
                What can I help with?
            </p>
            <p style="font-size:0.9rem; color:rgba(255,255,255,0.35);
                      margin:0; max-width:400px; line-height:1.75;">
                Ask me anything about the universe — a star's life cycle, the latest
                solar storm, an exoplanet's atmosphere, or what the rover saw on
                Mars today.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Suggestion prompt cards ────────────────────────────────────────────
    c1, c2 = st.columns(2, gap="small")
    for i, (col, prompt) in enumerate(zip([c1, c2, c1, c2], _SUGGESTIONS)):
        with col:
            st.markdown('<div class="suggest-card">', unsafe_allow_html=True)
            if st.button(prompt, use_container_width=True, key=f"sug_{i}"):
                st.session_state.pending_prompt = prompt
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="chat-footer">WILL.AI · Qwen 3.7 + Live NASA Data</div>',
        unsafe_allow_html=True,
    )

else:
    # ── Render conversation history ────────────────────────────────────────
    for idx_m, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                msg_retrieved = message.get("retrieved_passages", [])
                if msg_retrieved:
                    with st.expander("Retrieved sources", expanded=False):
                        for p in msg_retrieved:
                            snippet = (p.get("snippet") or "").strip()
                            source = p.get("source") or "source"
                            doc = p.get("doc_id") or ""
                            if snippet:
                                st.markdown(f"**{source}** — {snippet}  \n_{doc}_")
                            else:
                                st.markdown(f"**{source}** — {doc}")

                msg_imgs = message.get("assets", {}).get("images", [])
                msg_data_ctx = message.get("assets", {}).get("data", {})
                msg_desc = message.get("content", "")
                if msg_imgs:
                    _cols = st.columns(min(len(msg_imgs), 3))
                    for i, (_col, img) in enumerate(zip(_cols, msg_imgs[:3])):
                        with _col:
                            _t = fetch_thumb(img.get("thumb_url") or img.get("url", ""))
                            st.image(_t, caption=img.get("caption", "")[:50])
                            _already = any(
                                item["url"] == img.get("url", "")
                                for item in st.session_state.video_queue
                            )
                            if st.button(
                                "✓ Queued" if _already else "📌 Add to queue",
                                key=f"q_hist_{idx_m}_{i}",
                                disabled=_already,
                                use_container_width=True,
                            ):
                                _queue_image(img, msg_desc, msg_data_ctx)
                                st.rerun()

    # ── Process new user input ────────────────────────────────────────────
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            qwen_client = QwenClient(api_key=QWEN_API_KEY, model=MODEL_CHAT)
            chat_agent = ChatAgent(qwen_client)
            history = _db.get_conversation_history(st.session_state.conversation_id)

            with st.chat_message("assistant"):
                result: dict = {}

                # Show a connecting indicator while waiting for the first token.
                # st.write_stream renders nothing until the API sends the first
                # chunk, which can take several seconds on cold starts.
                _thinking = st.empty()
                _thinking.markdown(
                    '<span style="color:rgba(255,255,255,0.35); font-size:0.875rem;">'
                    "Connecting to Qwen…</span>",
                    unsafe_allow_html=True,
                )

                _gen = chat_agent.answer_stream_internal(user_input, history, result)
                _seen_first = [False]

                def _stream():
                    for _chunk in _gen:
                        if not _seen_first[0]:
                            _thinking.empty()  # dismiss indicator on first token
                            _seen_first[0] = True
                        yield _chunk

                answer = st.write_stream(_stream())
                answer = answer if isinstance(answer, str) else "".join(answer)
                if not answer and result.get("answer"):
                    answer = result["answer"]

                retrieved = result.get("retrieved_passages", [])
                if retrieved:
                    with st.expander("Retrieved sources", expanded=False):
                        for p in retrieved:
                            snippet = (p.get("snippet") or "").strip()
                            source = p.get("source") or "source"
                            doc = p.get("doc_id") or ""
                            if snippet:
                                st.markdown(f"**{source}** — {snippet}  \n_{doc}_")
                            else:
                                st.markdown(f"**{source}** — {doc}")

                _turn_assets = result.get("chat_assets", {})

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "assets": _turn_assets,
                    "retrieved_passages": retrieved,
                })
                _save_chat_turn(user_input, answer)

            # Rerun so images/queue buttons render from the history loop with
            # stable keys (q_hist_N_i).  Avoids the "two clicks" bug where the
            # first click fires against a q_live_* key that no longer exists.
            st.rerun()

        except (AuthenticationError, APIError) as exc:
            st.error(f"**Chat error:** {exc}")
        except Exception as exc:
            st.error(f"**Unexpected error:** {exc}")

    st.markdown(
        '<div class="chat-footer">WILL.AI · Qwen 3.7 + Live NASA Data</div>',
        unsafe_allow_html=True,
    )
