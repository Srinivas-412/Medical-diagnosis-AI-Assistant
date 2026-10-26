"""Streamlit chat UI for the Medical Diagnosis Chatbot."""
import uuid
import streamlit as st
from agents.medical_agent import run_agent

st.set_page_config(page_title="Medical Diagnosis AI", page_icon="🩺", layout="wide")

# ── Session state ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🩺 Medical AI Assistant")
    st.caption("Powered by PubMed · Groq · LangGraph · RAG")
    st.divider()
    st.markdown(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
    if st.button("🗑 New Conversation"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.warning(
        "⚠️ This tool is for informational purposes only. "
        "Always consult a licensed medical professional."
    )

# ── Chat header ────────────────────────────────────────────────────────────────
st.title("Medical Diagnosis AI Assistant")
st.caption("Ask about diseases, symptoms, diagnostic tests, and treatments — grounded in PubMed literature.")

# ── Render history ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            with st.expander("ℹ️ Response metadata"):
                st.json(msg["meta"])

# ── Input ──────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Describe symptoms or ask a medical question…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Searching PubMed and generating answer…"):
            result = run_agent(
                session_id=st.session_state.session_id,
                question=prompt,
            )

        answer = result["final_answer"]
        st.markdown(answer)

        meta = {
            "cache_hit": result["cache_hit"],
            "retry_count": result["retry_count"],
            "sources_count": len(result.get("sources", [])),
        }
        with st.expander("ℹ️ Response metadata"):
            st.json(meta)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "meta": meta,
    })
