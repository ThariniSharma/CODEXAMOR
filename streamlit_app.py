"""
Codexamor — Streamlit Local Runner (Python 3.14 / VS Code / Google Colab)
Run in VS Code: streamlit run streamlit_app.py
"""

import os
import streamlit as st  # type: ignore[import-not-found]

try:
    from google.genai import Client  # type: ignore[import-not-found]
    from google.genai import types  # type: ignore[import-not-found]
except ImportError:
    Client = None
    types = None

from federated_nlp_core import HybridLocalRAG, init_sqlite, redact_pii
from flower_simulation import run_scalable_simulation

st.set_page_config(page_title="Codexamor — Federated NLP", layout="wide")
st.markdown(
    """
    <style>
      .stApp { background-color: #08070B; color: #E6E3F0; }
      .stButton>button { background-color: #7C3AED; color: white; border: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

conn = init_sqlite()


@st.cache_resource
def get_rag_engine() -> HybridLocalRAG:
    rag = HybridLocalRAG()
    rag.index_documents([
        {
            "title": "Local Differential Privacy Notes",
            "content": "FedAvg with L2 norm clipping and Gaussian noise preserves client confidentiality.",
        }
    ])
    return rag


st.title("Codexamor — Privacy-Preserving Federated NLP Assistant")
tab_search, tab_fl, tab_about = st.tabs(["Perplexity Search & RAG", "Federated Network", "About Project"])

with tab_search:
    query = st.text_area("Ask anything (local PII is masked before web grounding):")
    use_shield = st.toggle("Enable Zero-Trust PII Masking", value=True)
    if st.button("Search & Synthesize") and query.strip():
        sanitized, redactions = redact_pii(query) if use_shield else (query, [])
        if redactions:
            st.info(f"Masked {len(redactions)} sensitive entity(s) locally: {sanitized}")

        rag = get_rag_engine()
        local_hits = rag.search(sanitized, top_k=2)

        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key and Client is not None and types is not None:
            client = Client(api_key=api_key)
            resp = client.models.generate_content(
                model="gemini-4-hydrogen",
                contents=f"Query: {sanitized}\nLocal RAG Context: {local_hits}",
                config=types.GenerateContentConfig(tools=[{"google_search": {}}]),
            )
            output_text = resp.text or ""
        else:
            output_text = "\n\n".join(
                f"**{hit.title}** (score: {hit.score:.3f})\n\n{hit.excerpt}" for hit in local_hits
            )
            st.caption("Running in offline Local RAG mode (set GEMINI_API_KEY to enable live web search).")

        st.markdown(output_text)
        conn.execute(
            "INSERT INTO threads (query_sanitized, response_text, epsilon_spent) VALUES (?, ?, ?)",
            (sanitized, output_text, 0.25),
        )
        conn.commit()

with tab_fl:
    n_clients = st.slider("Public Domain Federated Clients", min_value=5, max_value=500, value=25)
    n_rounds = st.number_input("FedAvg Rounds", min_value=1, max_value=20, value=3)
    if st.button("Run Flower FedAvg Simulation"):
        run_scalable_simulation(num_clients=int(n_clients), num_rounds=int(n_rounds))
        st.success(f"Aggregated {n_rounds} rounds across {n_clients} public nodes with Differential Privacy.")

with tab_about:
    st.subheader("Team codexamor — B.Tech Final Year Project")
    st.write("Python 3.14 · PyTorch · Flower · Scikit-learn · Pandas · NumPy · Sentence Transformers · SQLite · Differential Privacy · RAG · Streamlit")
