import html

import requests
import streamlit as st

import rag_core as rc

WIKI_PDF_URL = "https://en.wikipedia.org/api/rest_v1/page/pdf/Machine_learning"
GITHUB_URL = "https://github.com/Shabbir12-cpu/rag-wikipedia-hybrid-retrieval"
EXAMPLES = [
    "Who coined the term machine learning and when?",
    "What is reinforcement learning?",
    "What is a Bayesian network?",
    "How do machine learning and statistics differ?",
    "How many parameters does GPT-4 have?",
]

def esc(text: str) -> str:
    """Escape text for st.markdown with HTML: no tags, no LaTeX from '$'."""
    return html.escape(text).replace("$", "&#36;")


st.set_page_config(page_title="RAG over Wikipedia", page_icon="📚", layout="centered")


# ------------------------------------------------------------------ cached resources
@st.cache_resource(show_spinner="Loading the embedding model…")
def get_embedder():
    return rc.load_embedder()


@st.cache_resource(show_spinner="Loading the language model (first start downloads ~3 GB)…")
def get_generator():
    return rc.load_generator()


@st.cache_data(show_spinner="Downloading the Wikipedia article…", ttl=24 * 3600)
def get_wikipedia_pdf():
    r = requests.get(WIKI_PDF_URL, headers={"User-Agent": "RAG-Portfolio-Demo/1.0 (educational use)"}, timeout=60)
    r.raise_for_status()
    return r.content


@st.cache_resource(show_spinner="Reading and indexing the document…", max_entries=3)
def get_retriever(pdf_bytes: bytes):
    pages = rc.extract_pages(pdf_bytes)
    chunks = rc.chunk_pages(pages)
    if not chunks:
        return None, 0, 0
    return rc.HybridRetriever(chunks, get_embedder()), len(pages), len(chunks)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Document")
    source = st.radio("Ask questions about", ["Wikipedia: Machine learning", "My own PDF"], label_visibility="collapsed")
    uploaded = None
    if source == "My own PDF":
        uploaded = st.file_uploader("Upload a text-based PDF", type=["pdf"])

    st.header("Retrieval settings")
    alpha = st.slider("Dense weight (alpha)", 0.0, 1.0, 0.2, 0.1,
                      help="0 = keyword search (BM25) only, 1 = embedding search only. 0.2 scored best in evaluation.")
    top_k = st.slider("Passages given to the model", 2, 6, 4)

    st.divider()
    st.markdown(
        "**How it works:** hybrid retrieval (BM25 + bge-small/FAISS) finds the most relevant passages, "
        "Qwen2.5-1.5B-Instruct answers only from them, and each sentence is matched back to the passage and page it came from.\n\n"
        "**Evaluation (35 labelled questions):** 85.7% Hit@1 retrieval, 97% key-fact accuracy, "
        "10/10 out-of-scope questions refused.\n\n"
        f"[Code and full evaluation on GitHub]({GITHUB_URL})"
    )

# ------------------------------------------------------------------ main
st.title("📚 Ask a document")
st.caption("Retrieval-augmented question answering with page citations. Answers come only from the document; "
           "if it does not contain the answer, the model says so.")

if source == "My own PDF":
    if uploaded is None:
        st.info("Upload a PDF in the sidebar to start.")
        st.stop()
    pdf_bytes, doc_name = uploaded.getvalue(), uploaded.name
else:
    try:
        pdf_bytes, doc_name = get_wikipedia_pdf(), "Wikipedia: Machine learning"
    except Exception as e:
        st.error(f"Could not download the Wikipedia article ({e}). Upload a PDF in the sidebar instead.")
        st.stop()

retriever, n_pages, n_chunks = get_retriever(pdf_bytes)
if retriever is None:
    st.error("No text could be extracted from this PDF. Scanned PDFs (images of text) are not supported.")
    st.stop()
st.caption(f"**{doc_name}** · {n_pages} pages · {n_chunks} passages indexed")

if source != "My own PDF":
    st.write("Try an example:")
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, use_container_width=True):
            st.session_state["question"] = ex

question = st.text_input("Your question", key="question", placeholder="e.g. What is supervised learning?")

if question:
    tok, model, device = get_generator()
    with st.spinner("Retrieving passages and writing the answer… (about 15–40 seconds on a free CPU)"):
        result = rc.answer(question, retriever, tok, model, device, k=top_k, alpha=alpha)

    st.subheader("Answer")
    retrieved = result["retrieved"]
    if result["refused"]:
        st.warning(rc.REFUSAL_TEXT)
    else:
        parts, cited = [], set()
        for s in result["support"]:
            if s["supported"]:
                cited.add(s["passage"])
                parts.append(f"{esc(s['sentence'])} <sup><b>[{s['passage']}]</b></sup>")
            else:
                parts.append(f"<span title='No close match in the retrieved passages'>{esc(s['sentence'])} "
                             f"<sup>⚠️</sup></span>")
        st.markdown(" ".join(parts), unsafe_allow_html=True)
        if cited:
            st.caption("Sources: " + " · ".join(f"[{n}] {rc.page_label(retrieved[n - 1])}" for n in sorted(cited)))
        if any(not s["supported"] for s in result["support"]):
            st.caption("⚠️ This sentence did not closely match any retrieved passage and may not come from the document.")

    with st.expander(f"Retrieved passages ({len(retrieved)})"):
        for n, c in enumerate(retrieved, start=1):
            st.markdown(f"**[{n}] {rc.page_label(c)}** · fusion score {c['score']:.2f}")
            st.markdown(f"<div style='font-size:0.9rem'>{esc(c['text'])}</div>", unsafe_allow_html=True)

st.divider()
st.caption("Article text from Wikipedia, available under CC BY-SA 4.0. Built by Shabbir Kagalwala.")
