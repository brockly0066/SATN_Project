"""
SATN - Semantic-Aware Transformer Network
Abstractive Text Summarization + Automated Knowledge Extraction

100% local: no API keys. Models are downloaded once from the free Hugging Face
hub (no login/token needed) and then cached.

Pipeline
  1. Read PDF (column-aware, headers/footers/references/captions removed) / DOCX / TXT
  2. Sentence split + semantic embeddings (sentence-transformers)
  3. Semantic-aware salience scoring + redundancy-aware selection (MMR)
  4. Abstractive generation with a pre-trained Transformer (DistilBART)
       - research papers : abstract -> abstractive "In brief", + key points + conclusion
       - short documents : paragraph-by-paragraph summaries
       - other long docs : salient sentences -> chunked abstractive summary
  5. Knowledge extraction: entities, key phrases, relation triples
"""
import io
import os
import re
import statistics
from collections import Counter
from functools import lru_cache

import numpy as np


# --------------------------------------------------------------------------
# 1. Document reading and cleaning
# --------------------------------------------------------------------------
def _rows(words, tol=3):
    rows = []
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        if rows and abs(rows[-1]["top"] - w["top"]) <= tol:
            rows[-1]["words"].append(w)
        else:
            rows.append({"top": w["top"], "words": [w]})
    for r in rows:
        r["words"].sort(key=lambda w: w["x0"])
        r["bottom"] = max(w["bottom"] for w in r["words"])
    return rows


def _classify(row, mid, gap=9):
    """L = left column only, R = right column only, S = both columns on this
    row (clear gutter), F = full-width row."""
    ws = row["words"]
    if all(w["x1"] <= mid + 4 for w in ws):
        return "L"
    if all(w["x0"] >= mid - 4 for w in ws):
        return "R"
    for a, b in zip(ws, ws[1:]):
        if b["x0"] - a["x1"] >= gap and a["x1"] <= mid + 10 and b["x0"] >= mid - 10:
            return "S"
    return "F"


def _is_heading(s):
    return len(s.split()) <= 8 and s.upper() == s and any(c.isalpha() for c in s)


def _lines_to_paras(lines):
    """lines: [(top, bottom, text)] -> paragraphs (break on big gaps / headings)."""
    if not lines:
        return []
    gaps = [lines[i + 1][0] - lines[i][1] for i in range(len(lines) - 1)]
    pos = [g for g in gaps if g > 0]
    base = statistics.median(pos) if pos else 2
    paras, cur = [], [lines[0][2]]
    for i in range(1, len(lines)):
        g = lines[i][0] - lines[i - 1][1]
        t, prev = lines[i][2], lines[i - 1][2]
        if g > base * 1.8 + 1 or _is_heading(t) or _is_heading(prev):
            paras.append(" ".join(cur))
            cur = [t]
        else:
            cur.append(t)
    paras.append(" ".join(cur))
    return paras


def _pdf_paragraphs(source):
    """Column-aware text extraction (fixes two-column papers whose columns got
    interleaved, producing garbled words)."""
    import pdfplumber
    out = []
    with pdfplumber.open(source) as pdf:
        for page in pdf.pages:
            W, H = page.width, page.height
            words = [w for w in page.extract_words(x_tolerance=1.5, y_tolerance=2)
                     if w["top"] > 0.07 * H and w["bottom"] < 0.945 * H]   # drop header/footer
            mid = W / 2
            left, right, blocks = [], [], []

            def flush():
                nonlocal left, right
                blocks.extend(_lines_to_paras(left))
                blocks.extend(_lines_to_paras(right))
                left, right = [], []

            for r in _rows(words):
                kind = _classify(r, mid)
                txt = " ".join(w["text"] for w in r["words"])
                if kind == "F":
                    flush()
                    blocks.append((r["top"], r["bottom"], txt))
                elif kind == "S":
                    lw = [w["text"] for w in r["words"] if w["x1"] <= mid + 10]
                    rw = [w["text"] for w in r["words"] if w["x1"] > mid + 10]
                    left.append((r["top"], r["bottom"], " ".join(lw)))
                    right.append((r["top"], r["bottom"], " ".join(rw)))
                elif kind == "L":
                    left.append((r["top"], r["bottom"], txt))
                else:
                    right.append((r["top"], r["bottom"], txt))
            flush()

            run = []
            for b in blocks:
                if isinstance(b, tuple):
                    run.append(b)
                else:
                    if run:
                        out.extend(_lines_to_paras(run))
                        run = []
                    out.append(b)
            if run:
                out.extend(_lines_to_paras(run))
    return out


_REF_HEAD = re.compile(r"^(?:[IVX]+\.\s*|\d+\.?\s*)?(REFERENCES|References|BIBLIOGRAPHY|Bibliography)\s*$")
_DROP_START = re.compile(
    r"(?i)^(fig(?:ure)?\.?\s*[\dIVX]+|table\s*[\dIVX]+|received\s|digital object identifier|"
    r"this work (is|was) (licensed|supported)|corresponding author|the associate editor)")


def _drop_block(p):
    if _DROP_START.match(p) or "associate editor coordinating" in p.lower():
        return True
    letters = sum(c.isalpha() for c in p)
    digits = sum(c.isdigit() for c in p)
    return len(p) > 0 and (digits / len(p) > 0.35 or letters / len(p) < 0.5)


def _clean_text(text):
    text = re.sub(r"(?<=[a-z])- (?=[a-z])", "", text)                 # "gen- erates"
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"\[\d+(?:\s*[,\u2013\u2014-]\s*\d+)*\]", "", text)  # [12], [3-5]
    text = re.sub(r"https?://\S+|www\.\S+|\S+@\S+\.\S+", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = []
    for ln in text.split("\n"):
        ln = ln.strip()
        if ln and not _drop_block(ln):
            lines.append(ln)
    return "\n".join(lines)


def extract_text(file_bytes: bytes, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        paras = []
        for p in _pdf_paragraphs(io.BytesIO(file_bytes)):
            if _REF_HEAD.match(p.strip()):
                break                                   # stop at the reference list
            paras.append(p)
        text = "\n".join(paras)
    elif name.endswith(".docx"):
        import docx
        d = docx.Document(io.BytesIO(file_bytes))
        text = "\n".join(p.text for p in d.paragraphs)
    else:
        text = file_bytes.decode("utf-8", errors="ignore")
    return _clean_text(text)


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


# Smaller 6-layer DistilBART (~0.9 GB) so it fits Streamlit Cloud's free RAM.
# On a bigger machine:  SATN_MODEL=sshleifer/distilbart-cnn-12-6
MODEL_NAME = os.environ.get("SATN_MODEL", "sshleifer/distilbart-cnn-6-6")


@lru_cache(maxsize=1)
def get_summarizer():
    """Load the model directly (works on transformers v4 and v5; the old
    pipeline("summarization") was removed in transformers v5)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=True).eval()
    model.generation_config.max_length = None
    model.generation_config.min_length = 0
    return tok, model


def _generate(text, max_new, min_new):
    import torch
    tok, model = get_summarizer()
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=max_new, min_new_tokens=min_new,
                             num_beams=2, length_penalty=2.0,
                             no_repeat_ngram_size=3, early_stopping=True)
    out = tok.decode(ids[0], skip_special_tokens=True,
                     clean_up_tokenization_spaces=True).strip()
    out = re.sub(r"\s+([.,;:!?])", r"\1", out)
    if out and out[-1] not in ".!?" and any(p in out for p in ".!?"):
        out = out[: max(out.rfind("."), out.rfind("!"), out.rfind("?")) + 1]
    return out


# --------------------------------------------------------------------------
# 2-3. Sentences + semantic-aware salience
# --------------------------------------------------------------------------
_ABBR = ["et al.", "e.g.", "i.e.", "Fig.", "Figs.", "Eq.", "Eqs.", "vs.", "No.", "Dr.",
         "Mr.", "Mrs.", "Prof.", "Sec.", "approx.", "etc.", "cf."]


def _good_sentence(s):
    w = s.split()
    if not (6 <= len(w) <= 70):
        return False
    if sum(c.isalpha() or c == " " for c in s) / len(s) < 0.8:
        return False
    if not (s[0].isupper() or s[0].isdigit() or s[0] in "\"'(“"):
        return False
    if s.upper() == s or "©" in s or "doi" in s.lower():
        return False
    if re.match(r"(?i)^(index terms|keywords?)\b", s):
        return False
    return True


def split_sentences(text: str):
    sents = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        for a in _ABBR:
            para = para.replace(a, a.replace(".", "<DOT>"))
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(“])", para):
            s = s.replace("<DOT>", ".").strip()
            if _good_sentence(s):
                sents.append(s)
    return sents


def semantic_scores(sents, alpha=0.5, beta=0.4, gamma=0.1):
    """alpha: similarity to document centroid; beta: graph centrality (PageRank
    over sentence-sentence similarity); gamma: position prior."""
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

    return alpha * norm(sim_centroid) + beta * norm(central) + gamma * norm(pos), emb


def _mmr(sents, k, lam=0.7):
    """Pick k salient sentences with Maximal Marginal Relevance; keep original order."""
    if len(sents) <= k:
        return list(range(len(sents)))
    score, emb = semantic_scores(sents)
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


def select_salient(sents, ratio=0.3, max_sents=40):
    # short/medium documents keep every sentence so no section is lost
    if len(sents) <= 15:
        return list(range(len(sents)))
    k = max(10, min(max_sents, int(len(sents) * ratio)))
    return _mmr(sents, k)


# --------------------------------------------------------------------------
# 4. Abstractive generation strategies
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
    """Generic: chunked abstractive summary of the selected sentences."""
    cfg = {"short": (30, 90), "medium": (50, 150), "detailed": (80, 220)}[detail]
    parts = []
    for ch in _chunks(sentences):
        n_tok = int(len(ch.split()) * 1.3)
        mx = min(cfg[1], max(40, int(n_tok * 0.6)))
        mn = max(10, min(cfg[0], int(mx * 0.5)))
        parts.append(_generate(ch, mx, mn))
    joined = " ".join(parts)
    if len(parts) > 3:
        joined = _generate(joined, cfg[1] + 60, cfg[0])
    return joined


def summarize_by_paragraph(text, detail="medium"):
    """Short/medium documents: summarize every paragraph so each section appears."""
    paras = [p.strip() for p in re.split(r"\n+", text) if len(p.split()) >= 15]
    total = sum(len(p.split()) for p in paras)
    if not (2 <= len(paras) <= 12) or total > 1200:
        return None
    factor = {"short": 0.3, "medium": 0.45, "detailed": 0.65}[detail]
    out = []
    for p in paras:
        n_tok = int(len(p.split()) * 1.3)
        mx = min(150, max(30, int(n_tok * factor)))
        out.append(_generate(p, mx, max(10, int(mx * 0.5))))
    return " ".join(out)


# ---- research-paper mode ---------------------------------------------------
def split_sections(text):
    """Return (abstract, body, conclusion) using the paper's own headings."""
    abstract = conclusion = ""
    a_end, c_start = 0, len(text)

    m = re.search(r"(?ms)^\s*(?:ABSTRACT|Abstract)\b[\s:.\u2014-]*(.*?)"
                  r"(?=^\s*(?:INDEX TERMS|Index Terms|KEYWORDS|Keywords)\b"
                  r"|^\s*(?:[IVX]+\.|1\.?)?\s*(?:INTRODUCTION|Introduction)\b)", text[:6000])
    if m:
        abstract = m.group(1).strip()
        a_end = m.end()

    cs = list(re.finditer(r"(?m)^\s*(?:[IVX]+\.|\d+\.)?\s*(?:CONCLUSIONS?|Conclusions?)\b[^\n]*\n",
                          text))
    if cs and cs[-1].start() > len(text) * 0.5:
        c = cs[-1]
        tail = text[c.end():]
        cut = re.search(r"(?m)^\s*(?:ACKNOWLEDG|Acknowledg|APPENDIX|Appendix|REFERENCES)", tail)
        conclusion = (tail[: cut.start()] if cut else tail).strip()
        c_start = c.start()

    body = text[a_end:c_start]
    return abstract, body, conclusion


_DANGLING = re.compile(r"^(It|Its|This|These|Those|They|Such|He|She|However|Moreover|Therefore|"
                       r"Thus|Hence|Also|Furthermore|Additionally|In addition|Finally|Next|Then)\b")
_REFERS = re.compile(r"(?i)\b(fig(?:ure)?s?\.?|table|section|et al|eq\.|equation)\b")


def _standalone(s):
    """Sentences that still make sense when read on their own."""
    return not _DANGLING.match(s) and not _REFERS.search(s)


def structured_summary(text, detail="medium"):
    """Research papers: abstractive 'In brief' from the abstract + key points from
    the body + conclusion. Returns None when the text has no abstract."""
    abstract, body, conclusion = split_sections(text)
    if len(abstract.split()) < 40 or len(text.split()) < 900:
        return None

    cfg = {"short": (30, 80), "medium": (50, 120), "detailed": (70, 160)}[detail]
    brief = _generate(" ".join(split_sentences(abstract)) or abstract, cfg[1], cfg[0])
    parts = ["IN BRIEF\n" + brief]

    k = {"short": 4, "medium": 6, "detailed": 9}[detail]
    cand = [s for s in split_sentences(body) if _standalone(s)]
    if len(cand) >= 3:
        pts = [cand[i] for i in _mmr(cand, k)]
        parts.append("KEY POINTS\n" + "\n".join("• " + s for s in pts))

    cs = [s for s in split_sentences(conclusion)]
    if cs:
        keep = [cs[i] for i in _mmr(cs, 3)]
        parts.append("CONCLUSION\n" + " ".join(keep))
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# 5. Knowledge extraction
# --------------------------------------------------------------------------
def extract_knowledge(text: str, top_k=12):
    nlp = get_nlp()
    doc = nlp(text[:300_000])

    ents = Counter((e.text.strip(), e.label_) for e in doc.ents
                   if len(e.text.strip()) > 2 and e.label_ not in {"CARDINAL", "ORDINAL"})
    entities = [{"entity": t, "type": l, "count": c} for (t, l), c in ents.most_common(25)]

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
    text = _clean_text(text)
    sents = split_sentences(text)
    if len(sents) < 3:
        return {"error": "Not enough readable text to summarize."}
    salient = [sents[i] for i in select_salient(sents, ratio=ratio)]

    summary = structured_summary(text, detail)
    mode = "research paper (abstract + key points + conclusion)"
    if summary is None:
        summary = summarize_by_paragraph(text, detail)
        mode = "paragraph by paragraph"
    if summary is None:
        summary = abstractive_summary(salient, detail)
        mode = "salient sentences -> abstractive"
    return {
        "n_sentences": len(sents),
        "mode": mode,
        "salient_sentences": salient,
        "abstractive_summary": summary,
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
    print("\n=== SUMMARY (%s) ===\n%s" % (res["mode"], res["abstractive_summary"]))
    print("\n=== KEYPHRASES ===\n", ", ".join(res["knowledge"]["keyphrases"]))
