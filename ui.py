# ui.py — Streamlit chat UI for the Hybrid RAG API (api.py).

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Marketing Assistant", page_icon="📊")
st.title("📊 Marketing Assistant")



@st.cache_data(ttl=5)
def api_status():
    try:
        return requests.get(f"{API_URL}/health", timeout=3).json()
    except Exception:
        return {"status": "offline"}


if api_status().get("status") != "ok":
    st.error("🔴 API offline — start it with:  uvicorn api:app --port 8000")
    st.stop()
else:
    st.success("🟢 API connected")


def ask(question):
    resp = requests.post(f"{API_URL}/query", json={"question": question}, timeout=180)
    resp.raise_for_status()
    return resp.json()


st.subheader("Ask about Apple's financial statements")
question = st.text_input(
    "Your question",
    placeholder="e.g., What were Apple's total net sales for the three months ended December 30, 2023?",
)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Retrieving · reranking · generating…"):
        result = ask(question.strip())
    st.markdown("### Answer")
    st.write(result["answer"])
    st.markdown("### Sources")
    for src in result["sources"]:
        st.markdown(f"- {src}")
