"""
SATN - Semantic-Aware Transformer Network
Abstractive Text Summarization + Automated Knowledge Extraction

100% local: no API keys. Models are downloaded once from the free Hugging Face
hub (no login/token needed) and then cached on your machine.

Pipeline
  1. Read PDF / DOCX / TXT
  2. Sentence split + semantic embeddings (sentence-transformers)
  3. Semantic-aware salience scoring (centroid similarity + graph centrality
     + position) and redundancy-aware selection (MMR)   -> "semantic attention"
  4. Abstractive generation with a pre-trained Transformer (DistilBART)
  5. Knowledge extraction: entities, key phrases, relation triples
"""
import io
import re
from collections import Counter
from functools import lru_cache

import numpy as np

# --------------------------------------------------------------------------
# 1. Document reading
# --------------------------------------------------------------------------
def extract_text(file_bytes: bytes, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        import pdfplumber
        pages = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for p in pdf.pages:
                pages.append(p.extract_text() or "")
        text = "\n".join(pages)
    elif name.endswith(".docx"):
        import docx
        d = docx.Document(io.BytesIO(file_bytes))
        text = "\n".join(p.text for p in d.paragraphs)
    else:  # txt / md
        text = file_bytes.decode("utf-8", errors="ignore")
    text = re.sub(r"-\n(\w)", r"\1", text)        # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


# --------------------------------------------------------------------------
# Lazy model loaders (cached)
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_nlp():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError("Run:  python -m spacy download en_core_web_sm")


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def get_summarizer():
    from transformers import pipeline
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


# --------------------------------------------------------------------------
# 2-3. Semantic-aware salience (extractive guidance for the transformer)
# --------------------------------------------------------------------------
def split_sentences(text: str):
    doc = get_nlp()(text[:1_000_000])
    sents = [s.text.strip() for s in doc.sents if len(s.text.split()) >= 5]
    return sents


def semantic_scores(sents, alpha=0.5, beta=0.4, gamma=0.1):
    """Score each sentence by semantic importance.
       alpha: similarity to document centroid (global meaning)
       beta : graph centrality (sentence-sentence semantic links, PageRank)
       gamma: position prior (early sentences carry more information)"""
    import networkx as nx
    emb = get_embedder().encode(sents, normalize_embeddings=True)
    centroid = emb.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-9
    sim_centroid = emb @ centroid

    sim = emb @ emb.T
    np.fill_diagonal(sim, 0)
    sim[sim < 0.2] = 0
    g = nx.from_numpy_array(sim)
    pr = nx.pagerank(g, weight="weight") if len(sents) > 1 else {0: 1.0}
    central = np.array([pr[i] for i in range(len(sents))])

    pos = 1.0 / (1.0 + np.arange(len(sents)))

    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-9)

    score = alpha * norm(sim_centroid) + beta * norm(central) + gamma * norm(pos)
    return score, emb


def select_salient(sents, ratio=0.3, max_sents=40, lam=0.7):
    """Pick salient sentences with MMR to minimise redundancy,
       then restore original order for coherence."""
    if len(sents) <= 3:
        return list(range(len(sents)))
    score, emb = semantic_scores(sents)
    k = max(3, min(max_sents, int(len(sents) * ratio)))
    chosen, candidates = [], list(range(len(sents)))
    while candidates and len(chosen) < k:
        best, best_val = None, -1e9
        for i in candidates:
            red = max((emb[i] @ emb[j] for j in chosen), default=0)
            val = lam * score[i] - (1 - lam) * red
            if val > best_val:
                best, best_val = i, val
        chosen.append(best)
        candidates.remove(best)
    return sorted(chosen)


# --------------------------------------------------------------------------
# 4. Abstractive generation
# --------------------------------------------------------------------------
def _chunks(sentences, max_words=450):
    cur, n = [], 0
    for s in sentences:
        w = len(s.split())
        if cur and n + w > max_words:
            yield " ".join(cur)
            cur, n = [], 0
        cur.append(s)
        n += w
    if cur:
        yield " ".join(cur)


def abstractive_summary(sentences, detail="medium"):
    summ = get_summarizer()
    cfg = {"short": (40, 90), "medium": (60, 150), "detailed": (90, 220)}[detail]
    parts = []
    for ch in _chunks(sentences):
        n = len(ch.split())
        mx = min(cfg[1], max(30, int(n * 0.6)))
        mn = min(cfg[0], mx - 10)
        out = summ(ch, max_length=mx, min_length=max(10, mn),
                   do_sample=False, truncation=True)[0]["summary_text"]
        parts.append(out.strip())
    joined = " ".join(parts)
    # second pass if many chunks -> coherent single summary
    if len(parts) > 3:
        joined = summ(joined, max_length=cfg[1] + 60, min_length=cfg[0],
                      do_sample=False, truncation=True)[0]["summary_text"]
    return joined


# --------------------------------------------------------------------------
# 5. Knowledge extraction
# --------------------------------------------------------------------------
def extract_knowledge(text: str, top_k=12):
    nlp = get_nlp()
    doc = nlp(text[:300_000])

    # Named entities
    ents = Counter((e.text.strip(), e.label_) for e in doc.ents
                   if len(e.text.strip()) > 2 and e.label_ not in {"CARDINAL", "ORDINAL"})
    entities = [{"entity": t, "type": l, "count": c}
                for (t, l), c in ents.most_common(25)]

    # Key phrases: noun chunks ranked by semantic similarity to the document
    chunks = Counter(c.text.lower().strip() for c in doc.noun_chunks
                     if 1 <= len(c.text.split()) <= 4 and not c.root.is_stop
                     and c.root.pos_ in {"NOUN", "PROPN"})
    cands = [p for p, _ in chunks.most_common(150)]
    keyphrases = []
    if cands:
        e = get_embedder()
        doc_vec = e.encode([text[:5000]], normalize_embeddings=True)[0]
        vecs = e.encode(cands, normalize_embeddings=True)
        freq = np.array([chunks[p] for p in cands], dtype=float)
        score = (vecs @ doc_vec) + 0.1 * np.log1p(freq)
        keyphrases = [cands[i] for i in np.argsort(-score)[:top_k]]

    # Relations: subject - verb - object triples from dependency parse
    triples = []
    for sent in doc.sents:
        for tok in sent:
            if tok.pos_ == "VERB":
                subj = [c for c in tok.children if c.dep_ in ("nsubj", "nsubjpass")]
                obj = [c for c in tok.children if c.dep_ in ("dobj", "attr", "pobj")]
                if not obj:
                    for p in [c for c in tok.children if c.dep_ == "prep"]:
                        obj += [c for c in p.children if c.dep_ == "pobj"]
                if subj and obj:
                    s = " ".join(w.text for w in subj[0].subtree)
                    o = " ".join(w.text for w in obj[0].subtree)
                    if len(s.split()) <= 6 and len(o.split()) <= 6:
                        triples.append((s, tok.lemma_, o))
        if len(triples) >= 30:
            break
    return {"entities": entities, "keyphrases": keyphrases, "relations": triples[:20]}


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def run_satn(text: str, ratio=0.3, detail="medium"):
    sents = split_sentences(text)
    if len(sents) < 3:
        return {"error": "Document too short to summarize."}
    idx = select_salient(sents, ratio=ratio)
    salient = [sents[i] for i in idx]
    return {
        "n_sentences": len(sents),
        "salient_sentences": salient,                       # extractive view
        "abstractive_summary": abstractive_summary(salient, detail),
        "knowledge": extract_knowledge(text),
    }


def evaluate(candidate: str, reference: str):
    """ROUGE (offline) for when you have a reference summary."""
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    return {k: round(v.fmeasure, 4) for k, v in sc.score(reference, candidate).items()}


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    with open(path, "rb") as f:
        txt = extract_text(f.read(), path)
    res = run_satn(txt)
    print("\n=== SUMMARY ===\n", res["abstractive_summary"])
    print("\n=== KEYPHRASES ===\n", ", ".join(res["knowledge"]["keyphrases"]))
