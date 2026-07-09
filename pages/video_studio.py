"""Video Studio page — queue NASA images from chat and generate Wan 2.7 clips."""
from __future__ import annotations

import os
import queue
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

from agent.orchestrator import Orchestrator
from agent.run_db import Message
from utils.helpers import fetch_thumb
from utils.session import get_session_dir

# ── Environment ────────────────────────────────────────────────────────────────
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")
NASA_API_KEY = os.environ.get("NASA_API_KEY", "DEMO_KEY")

# Session state is already initialised by app.py
_session_dir = get_session_dir(st.session_state.session_id)

# ── Pipeline stage labels ──────────────────────────────────────────────────────
_LABELS = {
    "data":       "Fetching NASA data",
    "script":     "Writing scene captions",
    "storyboard": "Generating storyboard",
    "video":      "Generating video clips (Wan 2.7)",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _merge_data_contexts(queue_items: list[dict]) -> dict:
    """Merge data_context dicts from all queued items; later items win on collision."""
    merged: dict = {}
    for item in queue_items:
        merged.update(item.get("data_context") or {})
    return merged


def _render_pipeline_status(updates: list[dict], running: bool) -> dict:
    """Render accumulated pipeline updates inside a st.status block. Returns manifest."""
    manifest: dict = {}
    last_per_stage: dict[str, dict] = {}
    warnings_list: list[str] = []
    error_detail: str = ""

    for upd in updates:
        stage = upd.get("stage", "")
        if stage == "done":
            manifest = upd.get("manifest", {})
        elif stage == "warning":
            warnings_list.append(upd.get("detail", ""))
        elif stage == "error":
            error_detail = upd.get("detail", "")
        else:
            last_per_stage[stage] = upd

    has_error = bool(error_detail)
    if has_error:
        overall_state, overall_label = "error", "Pipeline error"
    elif not running:
        overall_state, overall_label = "complete", "Pipeline complete."
    else:
        running_stages = [s for s, u in last_per_stage.items() if u.get("status") == "running"]
        lbl = _LABELS.get(running_stages[-1], "Running…") if running_stages else "Running…"
        overall_state, overall_label = "running", f"{lbl}…"

    with st.status(overall_label, state=overall_state, expanded=True):
        for stage, upd in last_per_stage.items():
            label = _LABELS.get(stage, stage.title())
            if upd.get("status") == "running":
                st.markdown(f"**{label}** &nbsp; *running…*")
            else:
                st.markdown(f"**{label}** &nbsp; {upd.get('detail', '')}")
        for w in warnings_list:
            st.warning(w)
        if has_error:
            if "AllocationQuota" in error_detail or "FreeTier" in error_detail:
                st.error(
                    "**Video generation quota exhausted.** "
                    "Open the [DashScope console](https://dashscope.console.aliyun.com/) "
                    "and disable **\"Use free tier only\"**."
                )
            else:
                st.error(f"**Pipeline error:** {error_detail}")

    return manifest


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    if st.button("← Back to Chat", use_container_width=True, key="back_to_chat", type="primary"):
        st.switch_page("pages/chat.py")

    st.markdown('<hr>', unsafe_allow_html=True)

    _q = st.session_state.video_queue
    if _q:
        st.markdown(
            f'<div style="font-size:0.8rem; color:rgba(255,255,255,0.55); padding:4px 4px 8px;">'
            f'{len(_q)} image{"s" if len(_q) != 1 else ""} queued</div>',
            unsafe_allow_html=True,
        )
        for item in _q[:6]:
            st.caption(f"• {(item.get('caption') or item.get('source') or 'NASA image')[:38]}")
        if len(_q) > 6:
            st.caption(f"… and {len(_q) - 6} more")
    else:
        st.caption("Queue is empty.")
        st.caption("Go to chat and click 📌 on NASA images to build your queue.")

    st.markdown('<hr style="margin-top:auto;">', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:10px; color:rgba(255,255,255,0.25); padding:4px;">WILL.AI · Video Studio</div>',
        unsafe_allow_html=True,
    )

# ── Page title ─────────────────────────────────────────────────────────────────
st.markdown("## 🎬 Video Studio")
st.markdown(
    '<div style="font-size:0.875rem; color:rgba(255,255,255,0.4); margin-bottom:1.5rem;">'
    "Queue NASA images from chat, write a prompt, and generate a Wan 2.7 cinematic clip."
    "</div>",
    unsafe_allow_html=True,
)

# ── Two-column layout ──────────────────────────────────────────────────────────
col_left, col_right = st.columns([2, 3], gap="large")

# ── LEFT: Queue + controls ─────────────────────────────────────────────────────
with col_left:
    _q = st.session_state.video_queue

    st.markdown("#### Queue")

    if _q:
        for _qi, _item in enumerate(_q):
            c_img, c_info, c_btn = st.columns([1, 5, 1])
            with c_img:
                _thumb_url = _item.get("thumb_url") or _item.get("url", "")
                if _thumb_url:
                    try:
                        st.image(fetch_thumb(_thumb_url), width=42)
                    except Exception:
                        st.markdown("🖼", unsafe_allow_html=True)
            with c_info:
                st.caption((_item.get("caption") or _item.get("source") or "NASA image")[:55])
            with c_btn:
                if st.button("✕", key=f"rm_{_qi}", help="Remove from queue"):
                    st.session_state.video_queue.pop(_qi)
                    st.rerun()

        if st.button("Clear queue", use_container_width=True):
            st.session_state.video_queue = []
            if "studio_prompt" in st.session_state:
                del st.session_state["studio_prompt"]
            st.rerun()
    else:
        st.markdown(
            '<div style="color:rgba(255,255,255,0.3); font-size:0.875rem; '
            'padding:1rem 0; text-align:center;">No images queued yet.</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    _video_prompt = st.text_area(
        "Video prompt",
        placeholder="Describe the video you want to generate…",
        height=120,
        key="studio_prompt",
    )

    _mode = st.radio(
        "Mode",
        ["I2V — use queued images", "T2V — text only"],
        index=0 if _q else 1,
        horizontal=True,
        help="I2V uses queued NASA images as reference frames. T2V fetches fresh NASA data.",
    )
    _use_i2v = _mode.startswith("I2V")

    _generate_clicked = st.button(
        "Generate Video",
        type="primary",
        use_container_width=True,
        disabled=not (_video_prompt or "").strip() or st.session_state._pipeline_running,
    )

    # ── Launch pipeline ────────────────────────────────────────────────────
    if _generate_clicked and (_video_prompt or "").strip() and not st.session_state._pipeline_running:
        _prompt_text = _video_prompt.strip()

        if _use_i2v and _q:
            _merged_assets: dict = {
                "query": _prompt_text,
                "images": [
                    {
                        "url": i["url"],
                        "thumb_url": i["thumb_url"],
                        "caption": i["caption"],
                        "source": i["source"],
                    }
                    for i in _q
                ],
                "data": _merge_data_contexts(_q),
                "tools_called": [],
            }
            _chat_desc = " ".join(i.get("description", "") for i in _q)[:400]
        else:
            # T2V: fetch NASA data first (synchronous, in main thread)
            _orch_fetch = Orchestrator(
                qwen_api_key=QWEN_API_KEY,
                nasa_api_key=NASA_API_KEY,
                output_dir=_session_dir,
            )
            _merged_assets = {}
            _fetch_status = st.status("Fetching NASA data…", expanded=True)
            try:
                for _upd in _orch_fetch.fetch_data(_prompt_text):
                    _merged_assets = _upd.get("assets", _merged_assets)
                    if _upd["status"] == "running":
                        with _fetch_status:
                            st.markdown(_upd["detail"])
                    elif _upd["status"] == "done":
                        _fetch_status.update(label=_upd["detail"], state="complete")
            except Exception as _exc:
                _fetch_status.update(label="Data fetch failed", state="error")
                st.error(f"**Error fetching NASA data:** {_exc}")
                st.stop()
            _chat_desc = ""

        # Persist context for post-pipeline render
        st.session_state._pipeline_prompt = _prompt_text
        st.session_state._pipeline_chat_desc = _chat_desc
        st.session_state._pipeline_merged_assets = _merged_assets
        st.session_state._pipeline_run_saved = False

        _cancel_ev = threading.Event()
        _pq: queue.Queue = queue.Queue()
        st.session_state._pipeline_cancel = _cancel_ev
        st.session_state._pipeline_queue = _pq
        st.session_state._pipeline_running = True
        st.session_state._pipeline_updates = []

        def _bg_pipeline(
            _p=_prompt_text,
            _a=_merged_assets,
            _d=_chat_desc,
            _q=_pq,
            _ev=_cancel_ev,
            _out=_session_dir,
        ) -> None:
            _orch = Orchestrator(
                qwen_api_key=QWEN_API_KEY,
                nasa_api_key=NASA_API_KEY,
                cancel_event=_ev,
                output_dir=_out,
            )
            try:
                for upd in _orch.run_pipeline(_p, _a, chat_description=_d):
                    _q.put(upd)
            except InterruptedError:
                _q.put({"stage": "warning", "status": "warning",
                        "detail": "Generation cancelled by user."})
            except Exception as exc:
                detail = str(exc)
                if "AllocationQuota" in detail or "FreeTier" in detail:
                    detail = (
                        "Video generation quota exhausted. "
                        "Disable \"Use free tier only\" in the DashScope console."
                    )
                _q.put({"stage": "error", "status": "error", "detail": detail})
            finally:
                _q.put(None)  # sentinel

        threading.Thread(target=_bg_pipeline, daemon=True).start()
        st.rerun()

# ── RIGHT: Pipeline status + video player ──────────────────────────────────────
with col_right:
    st.markdown("#### Status")

    # Drain pipeline queue on every re-render
    _live_pq = st.session_state._pipeline_queue
    if _live_pq is not None:
        try:
            while True:
                _item = _live_pq.get_nowait()
                if _item is None:  # sentinel: thread finished
                    st.session_state._pipeline_running = False
                    st.session_state._pipeline_queue = None
                    break
                st.session_state._pipeline_updates.append(_item)
        except queue.Empty:
            pass

    if st.session_state._pipeline_updates or st.session_state._pipeline_running:
        _manifest = _render_pipeline_status(
            st.session_state._pipeline_updates,
            st.session_state._pipeline_running,
        )

        if st.session_state._pipeline_running:
            if st.button("✕  Cancel", key="cancel_pipeline"):
                _ev = st.session_state._pipeline_cancel
                if _ev:
                    _ev.set()
            time.sleep(2)  # poll interval
            st.rerun()

        elif _manifest:
            # Show generated clips
            _clips_dir = _session_dir / "clips"
            _manifest_clips = [
                Path(p) for p in _manifest.get("clips", []) if Path(p).exists()
            ]
            _scene_clips = _manifest_clips or (
                sorted(_clips_dir.glob("scene_*.mp4"), key=lambda p: p.stat().st_mtime)
                if _clips_dir.exists()
                else []
            )
            for _clip in _scene_clips:
                st.caption(_clip.name)
                st.video(str(_clip))
                st.markdown('<div style="margin-bottom:1.5rem"></div>', unsafe_allow_html=True)

            # Save pipeline run to RunDB (once)
            if not st.session_state._pipeline_run_saved:
                _db = st.session_state.run_db
                _db.save_run(
                    run_id=str(uuid.uuid4()),
                    conversation_id=st.session_state.conversation_id,
                    user_message=st.session_state._pipeline_prompt,
                    assistant_response=(
                        f"{len(_scene_clips)} clip(s) generated."
                        if _scene_clips else "Pipeline complete."
                    ),
                    assets=st.session_state._pipeline_merged_assets,
                    manifest=_manifest,
                    messages=[
                        Message(
                            role=m["role"],
                            content=m["content"],
                            timestamp=datetime.now().isoformat(),
                        )
                        for m in st.session_state.messages
                    ],
                )
                st.session_state._pipeline_run_saved = True
    else:
        st.markdown(
            '<div style="color:rgba(255,255,255,0.25); font-size:0.875rem; '
            'padding:2rem 0; text-align:center;">Pipeline status will appear here.</div>',
            unsafe_allow_html=True,
        )
