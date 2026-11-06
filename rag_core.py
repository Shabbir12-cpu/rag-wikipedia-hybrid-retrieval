"""Core RAG pipeline: PDF ingestion, sentence-aware chunking, hybrid retrieval,
grounded generation and sentence-level citation attribution.

Same methods as the evaluation notebook (RAG_Wikipedia_Hybrid_Retrieval.ipynb).
"""
import os
import re

import faiss
import numpy as np
import pymupdf
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
GEN_MODEL = os.getenv("GEN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
REFUSAL_TEXT = "I don't know based on the provided document."
ATTR_THRESHOLD = 0.80

# ---------------------------------------------------------------- ingestion
END_HEADINGS = {"see also", "references", "notes", "further reading", "external links", "sources", "bibliography"}
CITATION_RE = re.compile(
    r"\[\s*(?:\d+(?:\s*[,–-]\s*\d+)*|[a-z]|citation needed|clarification needed|note \d+)\s*\]",
    flags=re.IGNORECASE,
)


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("­", "")
    text = CITATION_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:)])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    return text.strip()


def extract_pages(pdf_bytes: bytes, cut_back_matter: bool = True):
    """Return [{"page": n, "text": ...}] for each page, stopping at See also / References."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    min_page_for_cut = int(len(doc) * 0.3)
    pages = []
    for i, page in enumerate(doc):
        lines = page.get_text("text").splitlines()
        cut = False
        if cut_back_matter and i >= min_page_for_cut:
            for j, line in enumerate(lines):
                if line.strip().lower() in END_HEADINGS:
                    lines, cut = lines[:j], True
                    break
        text = clean_text(" ".join(lines))
        if text:
            pages.append({"page": i + 1, "text": text})
        if cut:
            break
    return pages


# ---------------------------------------------------------------- chunking
ABBREV_END = re.compile(r"\b(?:e\.g|i\.e|etc|vs|al|Dr|Mr|Ms|Fig|No|approx)\.$", flags=re.IGNORECASE)


def split_sentences(text: str):
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])", text)
    sents = []
    for p in parts:
        if sents and ABBREV_END.search(sents[-1]):
            sents[-1] += " " + p
        else:
            sents.append(p)
    return [s.strip() for s in sents if s.strip()]


def chunk_pages(pages, target_words=200, overlap_words=40, min_words=40):
    """Sentence-aware overlapping chunks that may cross page breaks and keep page numbers."""
    stream = [{"text": s, "page": p["page"], "n_words": len(s.split())}
              for p in pages for s in split_sentences(p["text"])]
    chunks, i, n = [], 0, len(stream)
    while i < n:
        j, words = i, 0
        while j < n and (words < target_words or j == i):
            words += stream[j]["n_words"]
            j += 1
        if words < min_words and chunks:
            prev = chunks[-1]
            new = stream[prev["sent_end"]:j]
            if new:
                prev["text"] += " " + " ".join(s["text"] for s in new)
                prev["page_end"] = new[-1]["page"]
                prev["sent_end"] = j
        else:
            sel = stream[i:j]
            chunks.append({"chunk_id": len(chunks), "text": " ".join(s["text"] for s in sel),
                           "page_start": sel[0]["page"], "page_end": sel[-1]["page"], "sent_end": j})
        if j >= n:
            break
        k, back = j, 0
        while k > i + 1 and back < overlap_words:
            k -= 1
            back += stream[k]["n_words"]
        i = k
    return chunks


# ---------------------------------------------------------------- retrieval
STOPWORDS = set("""a an and are as at be by for from has have in is it its of on or that the to was were what when
where which who whom why how with this these those do does did can into than then there their they he she we you i
about also been being but if not no so such""".split())


def bm25_tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def minmax(scores: dict) -> dict:
    if not scores:
        return {}
    v = np.array(list(scores.values()), dtype=float)
    lo, hi = v.min(), v.max()
    return {k: 1.0 for k in scores} if hi == lo else {k: (s - lo) / (hi - lo) for k, s in scores.items()}


class HybridRetriever:
    """Weighted fusion of dense (bge + FAISS) and BM25 scores. alpha = weight of the dense score."""

    def __init__(self, chunks, embedder):
        self.chunks = chunks
        self.texts = [c["text"] for c in chunks]
        self.embedder = embedder
        emb = embedder.encode(self.texts, batch_size=32, convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=False).astype("float32")
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        self.bm25 = BM25Okapi([bm25_tokenize(t) for t in self.texts])

    def dense(self, q, k):
        qe = self.embedder.encode([BGE_QUERY_PREFIX + q], convert_to_numpy=True,
                                  normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(qe, min(k, len(self.texts)))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def keyword(self, q, k):
        scores = self.bm25.get_scores(bm25_tokenize(q))
        return [(int(i), float(scores[i])) for i in np.argsort(scores)[::-1][:k]]

    def search(self, q, k=4, alpha=0.2, fetch_k=30):
        d = minmax(dict(self.dense(q, fetch_k)))
        b = minmax(dict(self.keyword(q, fetch_k)))
        fused = [(i, alpha * d.get(i, 0.0) + (1 - alpha) * b.get(i, 0.0)) for i in set(d) | set(b)]
        top = sorted(fused, key=lambda x: -x[1])[:k]
        return [{**self.chunks[i], "score": s} for i, s in top]


# ---------------------------------------------------------------- generation
SYSTEM_PROMPT = (
    "You answer questions using only the numbered context passages from the document.\n"
    "Rules:\n"
    "1. Use only facts stated in the context. Do not use outside knowledge.\n"
    "2. After each fact, cite the passage number(s) in square brackets, e.g. [1] or [2][3].\n"
    f"3. If the context does not contain the answer, reply exactly: {REFUSAL_TEXT}\n"
    "4. Answer in 1 to 3 sentences."
)


def load_embedder():
    return SentenceTransformer(EMBED_MODEL, device="cpu")


def load_generator(model_name=GEN_MODEL):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device).eval()
    return tok, model, device


def page_label(c):
    return f"page {c['page_start']}" if c["page_start"] == c["page_end"] else f"pages {c['page_start']}-{c['page_end']}"


def build_context(retrieved):
    return "\n\n".join(f"[{n}] ({page_label(c)}) {c['text']}" for n, c in enumerate(retrieved, start=1))


def generate(question, retrieved, tok, model, device, max_new_tokens=160):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{build_context(retrieved)}\n\nQuestion: {question}"},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             repetition_penalty=1.05, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------- citations
def is_refusal(ans):
    return bool(re.search(r"(i do not know|i don't know|i don’t know)", ans.lower()))


def strip_cites(ans):
    return re.sub(r"\s*\[\d+\]", "", ans).strip()


def attribute_citations(answer, retrieved, embedder, threshold=ATTR_THRESHOLD):
    """Match each answer sentence to the retrieved passage holding its most similar sentence."""
    sents = split_sentences(strip_cites(answer))
    ctx = [(n, cs) for n, c in enumerate(retrieved, start=1) for cs in split_sentences(c["text"])]
    if not sents or not ctx:
        return []
    A = embedder.encode(sents, convert_to_numpy=True, normalize_embeddings=True)
    C = embedder.encode([cs for _, cs in ctx], convert_to_numpy=True, normalize_embeddings=True)
    sims = A @ C.T
    result = []
    for i, s in enumerate(sents):
        j = int(sims[i].argmax())
        result.append({"sentence": s, "passage": ctx[j][0], "evidence": ctx[j][1],
                       "similarity": float(sims[i, j]), "supported": float(sims[i, j]) >= threshold})
    return result


def answer(question, retriever, tok, model, device, k=4, alpha=0.2):
    retrieved = retriever.search(question, k=k, alpha=alpha)
    raw = generate(question, retrieved, tok, model, device)
    support = [] if is_refusal(raw) else attribute_citations(raw, retrieved, retriever.embedder)
    return {"raw": raw, "refused": is_refusal(raw), "support": support, "retrieved": retrieved}
