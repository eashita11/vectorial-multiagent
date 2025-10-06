import streamlit as st
import time
from src.orchestrator import run_collaboration
from src.llm import warm_up

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="🎭 Multi-Agent Debate", layout="centered")

# Warm up the model once
warm_up("llama3")

# ---------------- COLORS ----------------
BACKGROUND = "#000000"
TEXT_COLOR = "#FFFFFF"
BOX_COLOR = "#1e1e1e"
COMMANDER_COLOR = "#1e88e5"
RATIONALIST_COLOR = "#2e7d32"
DRAMATIST_COLOR = "#8e24aa"
SYNTHESIS_COLOR = "#0d9488"

# ---------------- STYLES ----------------
st.markdown(
    f"""
    <style>
      .stApp {{
        background-color: {BACKGROUND};
        color: {TEXT_COLOR} !important;
      }}

      .block-container {{
        max-width: 900px;
        padding-top: 2rem !important;
      }}

      .bubble {{
        background: {BOX_COLOR};
        color: {TEXT_COLOR};
        border-radius: 14px;
        padding: 12px 16px;
        margin: 8px 0;
        border-left: 5px solid transparent;
        box-shadow: 0 2px 6px rgba(255,255,255,0.05);
      }}

      .user {{ border-left-color: #888; }}
      .commander {{ border-left-color: {COMMANDER_COLOR}; }}
      .rationalist {{ border-left-color: {RATIONALIST_COLOR}; }}
      .dramatist {{ border-left-color: {DRAMATIST_COLOR}; }}
      .synthesis {{ border-left-color: {SYNTHESIS_COLOR}; }}

      h3, h4 {{ color: {TEXT_COLOR}; }}
      .caption small {{ color: #aaa !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- TITLE ----------------
st.title("🎭 Multi-Agent Conversational Debate")
st.markdown(
    "Experience a live dialogue between **Commander**, **Rationalist**, and **Dramatist**, "
    "each offering unique perspectives — followed by a **consensus synthesis**."
)

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- CHAT INPUT ----------------
query = st.chat_input("Ask a question...")

# ---------------- RUN COLLAB ----------------
if query:
    with st.spinner("🤔 Agents are thinking…"):
        result = run_collaboration(query, dialogue_rounds=1)
    st.session_state.history.append((query, result))

# ---------------- TYPE EFFECT ----------------
def type_out(container, html, animate=True):
    if not animate:
        container.markdown(html, unsafe_allow_html=True)
        return
    ph = container.empty()
    acc = ""
    for ch in html:
        acc += ch
        ph.markdown(acc, unsafe_allow_html=True)
        time.sleep(0.006)

# ---------------- CLEAR HISTORY BUTTON ----------------
col_clear, _ = st.columns([1, 6])
with col_clear:
    if st.button("Clear history"):
        st.session_state.history = []

# ---------------- DISPLAY ----------------
for i, (q, res) in enumerate(st.session_state.history):
    is_last = (i == len(st.session_state.history) - 1)

    # ---------- USER ----------
    type_out(st, f"<div class='bubble user'><b>🧑 You:</b> {q}</div>", animate=is_last)

    # ---------- ROUND 1 ----------
    st.markdown("### 🗣️ Round 1 — Individual Viewpoints")
    round1 = res.get("round1", {})
    for key, label, css in [
        ("commander", "🧭 Commander", "commander"),
        ("rationalist", "🧠 Rationalist", "rationalist"),
        ("dramatist", "🎭 Dramatist", "dramatist"),
    ]:
        if key in round1:
            msg = round1[key].get("response", "")
            cits = round1[key].get("citations", "")
            html = f"<div class='bubble {css}'><b>{label}:</b> {msg}</div>"
            type_out(st, html, animate=is_last)
            if cits:
                st.caption(f"🔗 {cits}")

    # ---------- DIALOGUE ----------
    if res.get("dialogue"):
        st.markdown("### 💬 Live Dialogue — Agents Build on Each Other")
        for turn in res["dialogue"]:
            who = turn.get("speaker", "")
            msg = turn.get("message", "")
            css = (
                "commander"
                if "Commander" in who
                else "rationalist"
                if "Rationalist" in who
                else "dramatist"
            )
            st.markdown(
                f"<div class='bubble {css}'><b>{who}:</b> {msg}</div>",
                unsafe_allow_html=True,
            )
            if turn.get("citations"):
                st.caption(f"🔗 {turn['citations']}")

    # ---------- CHALLENGE ROUND ----------
    ch = res.get("challenges") or {}
    if ch:
        st.markdown("### ⚖️ Challenge Round — Agents Respond to Each Other")
        left, right = st.columns(2)

        with left:
            roc = ch.get("rationalist_on_commander")
            if roc:
                st.markdown("<h4>🧠 Rationalist → 🧭 Commander</h4>", unsafe_allow_html=True)
                st.markdown(f"<div class='bubble rationalist'>{roc.get('response','')}</div>", unsafe_allow_html=True)
                if roc.get("citations"):
                    st.caption(f"🔗 {roc['citations']}")

            rebut = ch.get("commander_rebuttal")
            if rebut:
                st.markdown("<h4>🧭 Commander (Rebuttal)</h4>", unsafe_allow_html=True)
                st.markdown(f"<div class='bubble commander'>{rebut.get('response','')}</div>", unsafe_allow_html=True)

        with right:
            rod = ch.get("rationalist_on_dramatist")
            if rod:
                st.markdown("<h4>🧠 Rationalist → 🎭 Dramatist</h4>", unsafe_allow_html=True)
                st.markdown(f"<div class='bubble rationalist'>{rod.get('response','')}</div>", unsafe_allow_html=True)
                if rod.get("citations"):
                    st.caption(f"🔗 {rod['citations']}")

            recon = ch.get("dramatist_reconcile")
            if recon:
                st.markdown("<h4>🎭 Dramatist (Reconcile)</h4>", unsafe_allow_html=True)
                st.markdown(f"<div class='bubble dramatist'>{recon.get('response','')}</div>", unsafe_allow_html=True)

    # ---------- SYNTHESIS ----------
    st.markdown("### 🧩 Consensus Summary — Unified Insight")
    final = res.get("synthesis")
    if isinstance(final, dict):
        st.markdown(
            f"<div class='bubble synthesis'><b>🧩 Synthesis:</b> {final.get('response','')}</div>",
            unsafe_allow_html=True,
        )
        if final.get("citations"):
            st.caption(f"🔗 {final['citations']}")
    elif final:
        st.markdown(
            f"<div class='bubble synthesis'><b>🧩 Synthesis:</b> {final}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")