"""Generate the single-notebook assignment deliverable."""

import json
from pathlib import Path

cells = []


def _cell(cell_type: str, source: str) -> dict:
    text = source.strip() + "\n"
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")[:-1]] + ([text.split("\n")[-1] + "\n"] if text.endswith("\n") else []),
    }
    # Keep a single-string source; Jupyter accepts both.
    cell["source"] = text
    if cell_type == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    return cell


def md(source: str) -> None:
    cells.append(_cell("markdown", source))


def code(source: str) -> None:
    cells.append(_cell("code", source))


md(
    """
# RAG Assignment: Baseline Retrieval vs. Cross-Encoder Re-Ranking

**Corpus:** *Magic: The Gathering Comprehensive Rules* (effective 19 September 2025)  
**Runtime:** Kaggle notebook, **GPU T4**, Internet **on**  
**Deliverable style:** one commented notebook covering ingest → index → two retrieval pipelines → stress tests → latency.

### Run on Kaggle
1. Create a new notebook. Set **Accelerator = GPU T4 x2** (or a single T4) and **Internet = On**.
2. File → Import this notebook, or paste the cells.
3. Add `mtg_comprehensive_rules.pdf` as a dataset or upload it into the notebook files (optional: the notebook will download it if missing).
4. **Run All**.

This notebook implements the two-stage retrieval contrast required by the assignment over **the same chunked corpus**:

1. **Pipeline 1 (Baseline):** Bi-Encoder dense retrieval → Top 10 → send **Top 3** to the LLM.
2. **Pipeline 2 (Re-Ranking):** same Top 10 → Cross-Encoder scores every `(query, chunk)` pair → send the **new Top 3** to the LLM with the **identical prompt**.

The Magic comprehensive rules are a good stress-test document: they mix near-identical vocabulary (`counter` as a noun vs. a verb), general rules plus later exceptions, and many `can't` / `doesn't` constraints that Bi-Encoders tend to smear together.
"""
)

md(
    """
## Architecture to be used

```
                    Magic Comprehensive Rules PDF
                                  |
                          text extraction
                                  |
                     rule-aware chunking (+ overlap)
                                  |
              Bi-Encoder (sentence-transformers MiniLM)
                                  |
                    FAISS IndexFlatIP (cosine via L2-norm)
                                  |
                         query embedding
                                  |
                    Top-10 cosine candidates
                         /                    \\
            Pipeline 1                        Pipeline 2
     take ranks 1–3 as-is              Cross-Encoder re-rank 10 pairs
                                              take new ranks 1–3
                         \\                    /
                      identical RAG prompt + same LLM
                         Answer 1              Answer 2
```

### Why two encoders?

| | **Bi-Encoder** | **Cross-Encoder** |
|---|---|---|
| How text is encoded | Query and chunk are embedded **independently** | Query and chunk are concatenated and scored **jointly** |
| Interaction | No token-level interaction at scoring time; only cosine / inner product of two vectors | Full self-attention across query tokens and chunk tokens |
| Speed | One vector per chunk, computed **offline**; query is one forward pass | One forward pass **per pair**; too slow for the whole corpus |
| Role here | Fast candidate generation (Top 10) | Precise re-ranking of those 10 pairs |

The two-stage design is the point of the assignment: Cross-Encoders are more accurate because they see the query and passage together, but they cannot be run over every chunk at query time. Bi-Encoder recall gets the right neighborhood; Cross-Encoder precision promotes the clause that actually answers the question.
"""
)

md(
    """
## 1. Setup and configuration

Models (open weights, run on the Kaggle GPU):

- **Bi-Encoder:** `sentence-transformers/all-MiniLM-L6-v2`
- **Cross-Encoder:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (named in the brief)
- **LLM:** `Qwen/Qwen2.5-1.5B-Instruct` in half precision on GPU — small enough for a T4, good enough for grounded RAG answers

Kaggle already ships PyTorch. We only add PDF parsing, sentence-transformers, and FAISS.
"""
)

code(
    """
# Kaggle: enable Internet + GPU, then run this cell first.
import sys
%pip -q install pypdf sentence-transformers faiss-cpu python-dotenv
print("python", sys.version)
"""
)

code(
    """
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pypdf import PdfReader
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm.auto import tqdm

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    for key in ("HF_TOKEN", "GOOGLE_API_KEY"):
        try:
            os.environ.setdefault(key, secrets.get_secret(key))
        except Exception:
            pass
except Exception:
    pass

ON_KAGGLE = Path("/kaggle/working").exists()
ROOT = Path("/kaggle/working") if ON_KAGGLE else Path.cwd()
if not ON_KAGGLE and not (ROOT / "data").exists() and (ROOT.parent / "data").exists():
    ROOT = ROOT.parent

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

PDF_URL = "https://media.wizards.com/2025/downloads/MagicCompRules%2020250919.pdf"
PDF_PATH = DATA_DIR / "mtg_comprehensive_rules.pdf"  # resolved in the next cell after searching uploads
BI_ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TOP_K_RETRIEVE = 10
TOP_K_CONTEXT = 3
MAX_CHUNK_CHARS = 1100
MIN_CHUNK_CHARS = 180
CHUNK_OVERLAP = 120
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("ROOT:", ROOT)
print("device:", DEVICE, "| cuda_name:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu")
print("torch", torch.__version__)
"""
)

code(
    """
def find_uploaded_rules_pdf() -> Path | None:
    \"\"\"Prefer a PDF you added (Kaggle dataset, /kaggle/working, or repo root) over a download.\"\"\"
    search_roots = [
        Path.cwd(),
        ROOT,
        DATA_DIR,
        Path(\"/kaggle/working\"),
        Path(\"/kaggle/input\"),
    ]
    skip_bits = (\"assignment\", \"lma\", \"rag_\")
    ranked: list[tuple[int, Path]] = []
    seen = set()
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob(\"*.pdf\"):
            if path in seen or \".venv\" in path.parts:
                continue
            seen.add(path)
            name = path.name.lower()
            if any(b in name for b in skip_bits):
                continue
            score = 0
            if \"mtg\" in name or \"magic\" in name or \"comprule\" in name or \"comprehensive\" in name:
                score += 5
            if path.stat().st_size > 500_000:
                score += 2
            if score:
                ranked.append((score, path))
    ranked.sort(key=lambda x: (-x[0], str(x[1])))
    return ranked[0][1] if ranked else None


def ensure_pdf(fallback: Path, url: str) -> Path:
    uploaded = find_uploaded_rules_pdf()
    if uploaded is not None:
        print(\"Using uploaded rules PDF:\", uploaded, uploaded.stat().st_size, \"bytes\")
        return uploaded
    if fallback.exists() and fallback.stat().st_size > 100_000:
        print(\"Using existing PDF:\", fallback, fallback.stat().st_size, \"bytes\")
        return fallback
    print(\"No upload found; downloading\", url)
    urllib.request.urlretrieve(url, fallback)
    print(\"Saved\", fallback, fallback.stat().st_size, \"bytes\")
    return fallback


PDF_PATH = ensure_pdf(PDF_PATH, PDF_URL)
print(\"PDF_PATH =\", PDF_PATH)
"""
)

md(
    """
## 2. Document choice

**Selected PDF:** Wizards of the Coast, *Magic: The Gathering Comprehensive Rules*, 19 September 2025.

Why this document (and not a short FAQ or Wikipedia article):

- It is a single, dense **rulebook**, which the assignment explicitly lists as a suitable class.
- Numbered subrules make gold-span analysis possible (`701.6a` vs `122.1`).
- It contains the three required failure types:
  - **Keyword trap:** `counter` means a marker on a permanent *and* the action of cancelling a spell.
  - **Exception / corner case:** the land-playing rules are later overridden by “you can’t play a land if it isn’t your turn; ignore effects that say otherwise.”
  - **Negation:** mana abilities “can’t be targeted, countered, or otherwise responded to.”
"""
)

code(
    """
def load_pdf_text(path: Path) -> str:
    \"\"\"Extract page text, drop page banners, and stitch hyphenated line breaks.\"\"\"
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(text)
    text = \"\\n\".join(pages)
    text = re.sub(r\"(?m)^\\s*\\d+\\s*$\", \"\", text)  # lone page numbers
    text = re.sub(r\"(\\w)-\\n(\\w)\", r\"\\1\\2\", text)
    text = re.sub(r\"\\n{3,}\", \"\\n\\n\", text)
    return text


raw_text = load_pdf_text(PDF_PATH)
print(f"Pages extracted via markers: {raw_text.count('[Page ')}")
print(f"Characters: {len(raw_text):,}")
print("Head of document:")
print(raw_text[:800])

# Confirm this copy still contains the gold clauses used in the stress tests.
needed = {
    "701.6a (counter a spell)": "701.6a",
    "305.3 (can't play a land off-turn)": "305.3.",
    "605.3b (mana abilities can't be responded to)": "605.3b",
    "111.6 (a token isn't a card)": "111.6.",
    "122.1 (counters as markers, keyword trap)": "122.1.",
}
print("\\nGold-clause check against this PDF:")
for label, needle in needed.items():
    idx = raw_text.find(needle)
    print(f\"  {'OK' if idx >= 0 else 'MISSING'}  {label}\")
    if idx >= 0:
        snippet = re.sub(r\"\\s+\", \" \", raw_text[idx:idx + 180])
        print(\"   \", snippet)
"""
)

md(
    """
## 3. Rule-aware chunking

Magic rules are already atomic (`122.1`, `122.1a`, `701.6a`). We split on those numbered headings, then pack adjacent tiny subrules so each chunk is large enough for MiniLM but still about one idea.

This is better than naive 512-token windows because a window could glue the *counters-as-markers* section to an unrelated nearby rule, or split an exception away from the general rule it qualifies.
"""
)

code(
    """
RULE_HEAD = re.compile(
    r\"(?m)^(?P<id>\\d{3}(?:\\.\\d+[a-z]*)?)\\.\\s+\"
)
HEADING_IDS = re.compile(r\"(?m)^(\\d{3}(?:\\.\\d+[a-z]*)?)\\.\\s+\")


def extract_rule_ids(text: str) -> list[str]:
    return HEADING_IDS.findall(text)


def parent_key(rule_id: str) -> str:
    \"\"\"305.2a and 305.2 share a parent; 305.3 starts a new group.\"\"\"
    m = re.match(r\"(\\d{3}(?:\\.\\d+)?)\", rule_id)
    return m.group(1) if m else rule_id


def split_numbered_rules(text: str) -> list[str]:
    \"\"\"Split on numbered headings. Skip the table of contents.\"\"\"
    first_sub = text.find(\"100.1.\")
    start = text.rfind(\"100. General\", 0, first_sub if first_sub != -1 else None)
    if start == -1:
        start = 0
    body = text[start:]
    matches = list(RULE_HEAD.finditer(body))
    blocks = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        block = body[m.start():end].strip()
        if block:
            blocks.append(block)
    return blocks


def pack_chunks(blocks: list[str], max_chars: int, min_chars: int) -> list[dict[str, Any]]:
    \"\"\"Pack only sibling subrules. Never cut a rule mid-word with character overlap.\"\"\"
    packed: list[str] = []
    buf = \"\"
    buf_key = None
    for block in blocks:
        rid = extract_rule_ids(block)
        key = parent_key(rid[0]) if rid else None
        if not buf:
            buf, buf_key = block, key
            continue
        same_family = key is not None and key == buf_key
        if same_family and (len(buf) < min_chars or len(buf) + 2 + len(block) <= max_chars):
            buf = buf + \"\\n\\n\" + block
        else:
            packed.append(buf)
            buf, buf_key = block, key
    if buf:
        packed.append(buf)

    chunks = []
    for i, text in enumerate(packed):
        ids = extract_rule_ids(text)
        chunks.append(
            {
                \"chunk_id\": i,
                \"rule_id\": ids[0] if ids else \"\",
                \"rule_ids\": ids,
                \"text\": text,
            }
        )
    for new_id, ch in enumerate(chunks):
        ch[\"row\"] = new_id
    return chunks


blocks = split_numbered_rules(raw_text)
chunks = pack_chunks(blocks, MAX_CHUNK_CHARS, MIN_CHUNK_CHARS)
print(\"Chunks missing a heading id:\", sum(1 for c in chunks if not c[\"rule_id\"]))
print(f\"Numbered blocks: {len(blocks):,}\")
print(f\"Packed chunks:   {len(chunks):,}\")
print(f\"Mean chunk chars: {np.mean([len(c['text']) for c in chunks]):.0f}\")
print(\"Sample chunk 0:\")
print(chunks[0][\"text\"][:500])
print(\"---\")
print(\"Sample later chunk:\")
print(chunks[min(80, len(chunks)-1)][\"text\"][:500])
"""
)

md(
    """
## 4. Pipeline 1 — Bi-Encoder index (dense retrieval)

Each chunk is mapped to a single vector **without seeing any query**. At query time we embed the query once and take inner products against L2-normalized chunk vectors, which is cosine similarity.

This is fast enough to search the whole corpus, but it cannot model word-sense (`counter`) or negation (`can't`) as well as a pairwise model.
"""
)

code(
    """
print(\"Loading Bi-Encoder on\", DEVICE, \":\", BI_ENCODER_NAME)
bi_encoder = SentenceTransformer(BI_ENCODER_NAME, device=DEVICE)

corpus_texts = [c[\"text\"] for c in chunks]
t0 = time.perf_counter()
corpus_embeddings = bi_encoder.encode(
    corpus_texts,
    batch_size=128 if DEVICE == \"cuda\" else 32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)
index_s = time.perf_counter() - t0

dim = corpus_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(corpus_embeddings.astype(\"float32\"))
print(f\"Indexed {index.ntotal} vectors of dim {dim} in {index_s:.1f}s\")


def baseline_retrieve(query: str, top_k: int = TOP_K_RETRIEVE) -> list[dict[str, Any]]:
    \"\"\"Return Top-k chunks by cosine similarity (Pipeline 1 candidate generator).\"\"\"
    q = bi_encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(\"float32\")
    scores, rows = index.search(q, top_k)
    hits = []
    for rank, (score, row) in enumerate(zip(scores[0], rows[0]), start=1):
        ch = chunks[int(row)]
        hits.append(
            {
                \"rank\": rank,
                \"row\": int(row),
                \"score\": float(score),
                \"rule_id\": ch[\"rule_id\"],
                \"rule_ids\": ch.get(\"rule_ids\", []),
                \"text\": ch[\"text\"],
            }
        )
    return hits
"""
)

md(
    """
## 5. Pipeline 2 — Cross-Encoder re-ranking

The Cross-Encoder reads `query [SEP] chunk` in one transformer. Attention can align “counter a spell” with “cancel it, removing it from the stack” and down-rank “a counter is a marker placed on an object.”

We **do not** re-embed the corpus with this model. We only score the 10 Bi-Encoder candidates, sort descending, and keep 3.
"""
)

code(
    """
print(\"Loading Cross-Encoder on\", DEVICE, \":\", CROSS_ENCODER_NAME)
cross_encoder = CrossEncoder(CROSS_ENCODER_NAME, device=DEVICE)


def rerank(query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    \"\"\"Score each (query, chunk) pair and return hits sorted by Cross-Encoder score.\"\"\"
    pairs = [(query, h[\"text\"]) for h in hits]
    ce_scores = cross_encoder.predict(pairs, convert_to_numpy=True)
    reranked = []
    for h, ce in zip(hits, ce_scores):
        item = dict(h)
        item[\"ce_score\"] = float(ce)
        item[\"baseline_rank\"] = h[\"rank\"]
        reranked.append(item)
    reranked.sort(key=lambda x: x[\"ce_score\"], reverse=True)
    for i, item in enumerate(reranked, start=1):
        item[\"rank\"] = i
    return reranked
"""
)

md(
    """
## 6. Shared LLM prompt

Both pipelines use this exact prompt template. The only difference is **which three chunks** are interpolated. That isolates retrieval quality as the cause of Answer 1 vs Answer 2.
"""
)

code(
    """
PROMPT_TEMPLATE = \"\"\"You are answering from the Magic: The Gathering Comprehensive Rules.

Use ONLY the retrieved rules below. If a later rule creates an exception, prefer the exception.
If the rules name a zone (graveyard, exile, hand), use that zone; do not invent one.
If the rules are insufficient, say you cannot tell from the retrieved text.
Quote the relevant rule numbers.

Retrieved rules:
{context}

Question: {question}

Answer:\"\"\"


def format_context(hits: list[dict[str, Any]], k: int = TOP_K_CONTEXT) -> str:
    parts = []
    for h in hits[:k]:
        parts.append(f\"[Rank {h['rank']} | rule {h.get('rule_id', '?')} | score={h.get('ce_score', h.get('score')):.4f}]\\n{h['text']}\")
    return \"\\n\\n---\\n\\n\".join(parts)


def build_prompt(question: str, hits: list[dict[str, Any]]) -> str:
    return PROMPT_TEMPLATE.format(context=format_context(hits), question=question)


print(\"Loading GPU LLM:\", LLM_NAME)
from transformers import AutoModelForCausalLM, AutoTokenizer

llm_tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
if llm_tokenizer.pad_token is None:
    llm_tokenizer.pad_token = llm_tokenizer.eos_token
llm_dtype = torch.float16 if DEVICE == \"cuda\" else torch.float32
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_NAME,
    torch_dtype=llm_dtype,
    device_map=\"auto\" if DEVICE == \"cuda\" else None,
)
if DEVICE != \"cuda\":
    llm_model = llm_model.to(DEVICE)
llm_model.eval()
print(\"LLM ready on\", next(llm_model.parameters()).device)


def generate_answer(prompt: str) -> str:
    \"\"\"Same generator for both pipelines. Grounding comes only from the retrieved chunks.\"\"\"
    messages = [
        {\"role\": \"system\", \"content\": \"Answer strictly from the provided Magic rules. Be concise.\"},
        {\"role\": \"user\", \"content\": prompt},
    ]
    # tokenize=False then encode: Transformers 5.x returns a BatchEncoding from
    # apply_chat_template(..., return_tensors='pt'), which has no .shape.
    chat_text = llm_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    encoded = llm_tokenizer(
        chat_text,
        return_tensors=\"pt\",
        truncation=True,
        max_length=3072,
    )
    encoded = {k: v.to(llm_model.device) for k, v in encoded.items()}
    input_len = encoded[\"input_ids\"].shape[-1]
    with torch.no_grad():
        out = llm_model.generate(
            **encoded,
            max_new_tokens=280,
            do_sample=False,
            pad_token_id=llm_tokenizer.eos_token_id,
        )
    new_tokens = out[0, input_len:]
    return llm_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
"""
)

md(
    """
## 7. End-to-end pipelines with latency timers

Retrieval time and generation time are recorded separately so the Cross-Encoder cost is visible and not mixed into LLM latency.
"""
)

code(
    """
@dataclass
class PipelineResult:
    name: str
    query: str
    hits: list[dict[str, Any]]
    context_hits: list[dict[str, Any]]
    prompt: str
    answer: str
    retrieve_ms: float
    rerank_ms: float
    generate_ms: float

    @property
    def retrieval_total_ms(self) -> float:
        return self.retrieve_ms + self.rerank_ms


def run_pipeline_1(query: str) -> PipelineResult:
    t0 = time.perf_counter()
    hits = baseline_retrieve(query, TOP_K_RETRIEVE)
    retrieve_ms = (time.perf_counter() - t0) * 1000
    context = hits[:TOP_K_CONTEXT]
    prompt = build_prompt(query, context)
    t1 = time.perf_counter()
    answer = generate_answer(prompt)
    generate_ms = (time.perf_counter() - t1) * 1000
    return PipelineResult(\"baseline\", query, hits, context, prompt, answer, retrieve_ms, 0.0, generate_ms)


def run_pipeline_2(query: str) -> PipelineResult:
    t0 = time.perf_counter()
    hits = baseline_retrieve(query, TOP_K_RETRIEVE)
    retrieve_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    hits = rerank(query, hits)
    rerank_ms = (time.perf_counter() - t1) * 1000
    context = hits[:TOP_K_CONTEXT]
    prompt = build_prompt(query, context)
    t2 = time.perf_counter()
    answer = generate_answer(prompt)
    generate_ms = (time.perf_counter() - t2) * 1000
    return PipelineResult(\"rerank\", query, hits, context, prompt, answer, retrieve_ms, rerank_ms, generate_ms)


def rank_of_rule(hits: list[dict[str, Any]], prefixes: list[str]) -> int | None:
    \"\"\"Rank of the first chunk whose *heading* is the gold rule, not a 'see rule X' citation.\"\"\"
    wanted = set(prefixes)
    for h in hits:
        heads = set(h.get(\"rule_ids\") or []) | set(extract_rule_ids(h[\"text\"]))
        if h.get(\"rule_id\"):
            heads.add(h[\"rule_id\"])
        for hid in heads:
            if hid in wanted or any(hid.startswith(p) for p in prefixes):
                return int(h[\"rank\"])
    return None
"""
)

md(
    """
## 8. Failure-mode queries (stress tests)

These queries were chosen **from this document**, not as generic RAG trivia. Each one has a distractor that a Bi-Encoder is likely to rank in the Top 3, and a gold clause that often sits in ranks 4–10 until the Cross-Encoder sees the pair.

| ID | Failure type | Query | Likely distractor | Gold rule |
|---|---|---|---|---|
| Q1 | Keyword trap | counter a spell | 122.x counters-as-markers | 701.6 counter-as-verb |
| Q2 | Exception / corner case | play a land on opponent's turn | 305.1 / 116.2a general land timing | 305.3 cannot, ignore effects |
| Q3 | Negation & constraints | respond to a mana ability | 605.3a when you *may* activate | 605.3b *can't* respond |
| Q4 | Document-specific / buried clause | destroyed token as a graveyard card | 404 graveyard / 701.7 create | 111.6–111.7 token isn't a card and ceases to exist |
"""
)

code(
    """
EVAL_QUERIES = [
    {
        \"id\": \"Q1_keyword_trap\",
        \"failure_type\": \"Keyword trap / distractor\",
        \"query\": (
            \"When you counter a spell, do you put a counter on that spell, and where does the spell card go?\"
        ),
        \"gold_prefixes\": [\"701.6\", \"701.6a\", \"701.6b\"],
        \"distractor_prefixes\": [\"122.1\", \"122.\"],
        \"why\": (
            \"The word 'counter' is dominated by +1/+1 counters (noun). The true answer is the keyword action "
            \"that cancels a spell on the stack and puts it into its owner's graveyard.\"
        ),
    },
    {
        \"id\": \"Q2_exception\",
        \"failure_type\": \"Exceptions & corner cases\",
        \"query\": (
            \"Can an effect instruct me to play a land during my opponent's turn, and does that work?\"
        ),
        \"gold_prefixes\": [\"305.3\"],
        \"distractor_prefixes\": [\"305.1\", \"116.2a\", \"305.2\"],
        \"why\": (
            \"The general land-play rule talks about main phases and priority. The later exception says a player "
            \"can't play a land if it isn't their turn and to ignore effects that instruct them to do so.\"
        ),
    },
    {
        \"id\": \"Q3_negation\",
        \"failure_type\": \"Negation & constraints\",
        \"query\": (
            \"Can an opponent respond to, target, or counter an activated mana ability?\"
        ),
        \"gold_prefixes\": [\"605.3b\"],
        \"distractor_prefixes\": [\"605.3a\", \"605.1\", \"117.1\"],
        \"why\": (
            \"Bi-Encoders often retrieve the positive permission ('you may activate whenever you have priority') "
            \"and miss the constraint that mana abilities do not use the stack and cannot be responded to.\"
        ),
    },
    {
        \"id\": \"Q4_buried_token\",
        \"failure_type\": \"Document-specific buried clause\",
        \"query\": (
            \"If a creature token is destroyed, is that token a card waiting in the graveyard?\"
        ),
        \"gold_prefixes\": [\"111.6\", \"111.7\", \"704.5d\"],
        \"distractor_prefixes\": [\"404.\", \"701.7\", \"111.1\"],
        \"why\": (
            \"Queries about 'destroyed' and 'graveyard' pull zone rules. The vital facts are that a token isn't a "
            \"card and that a token in any zone other than the battlefield ceases to exist.\"
        ),
    },
]


def preview_hits(title: str, hits: list[dict[str, Any]], n: int = 10) -> pd.DataFrame:
    rows = []
    for h in hits[:n]:
        preview = re.sub(r\"\\s+\", \" \", h[\"text\"])[:140]
        rows.append(
            {
                \"rank\": h[\"rank\"],
                \"baseline_rank\": h.get(\"baseline_rank\", h[\"rank\"]),
                \"rule_id\": h.get(\"rule_id\"),
                \"bi_score\": round(h.get(\"score\", float(\"nan\")), 4),
                \"ce_score\": round(h[\"ce_score\"], 4) if \"ce_score\" in h else None,
                \"preview\": preview,
            }
        )
    df = pd.DataFrame(rows)
    print(title)
    display(df)
    return df


all_runs: dict[str, dict[str, PipelineResult]] = {}
rank_rows = []
latency_rows = []

for item in EVAL_QUERIES:
    qid, query = item[\"id\"], item[\"query\"]
    print(\"\\n\" + \"=\" * 88)
    print(qid, \"|\", item[\"failure_type\"])
    print(\"Q:\", query)
    print(\"Why this should break baseline:\", item[\"why\"])

    p1 = run_pipeline_1(query)
    p2 = run_pipeline_2(query)
    all_runs[qid] = {\"p1\": p1, \"p2\": p2}

    preview_hits(\"Pipeline 1 — Top 10 Bi-Encoder\", p1.hits)
    preview_hits(\"Pipeline 2 — Top 10 after Cross-Encoder\", p2.hits)

    gold_p1 = rank_of_rule(p1.hits, item[\"gold_prefixes\"])
    gold_p2 = rank_of_rule(p2.hits, item[\"gold_prefixes\"])
    gold_in_p1_top3 = rank_of_rule(p1.context_hits, item[\"gold_prefixes\"]) is not None
    gold_in_p2_top3 = rank_of_rule(p2.context_hits, item[\"gold_prefixes\"]) is not None

    rank_rows.append(
        {
            \"query_id\": qid,
            \"failure_type\": item[\"failure_type\"],
            \"gold_rule\": \", \".join(item[\"gold_prefixes\"]),
            \"gold_rank_baseline\": gold_p1,
            \"gold_rank_rerank\": gold_p2,
            \"rank_shift\": (None if gold_p1 is None or gold_p2 is None else gold_p1 - gold_p2),
            \"gold_in_baseline_top3\": gold_in_p1_top3,
            \"gold_in_rerank_top3\": gold_in_p2_top3,
        }
    )
    for res in (p1, p2):
        latency_rows.append(
            {
                \"query_id\": qid,
                \"pipeline\": res.name,
                \"retrieve_ms\": round(res.retrieve_ms, 2),
                \"rerank_ms\": round(res.rerank_ms, 2),
                \"retrieval_total_ms\": round(res.retrieval_total_ms, 2),
                \"generate_ms\": round(res.generate_ms, 2),
            }
        )

    print(\"\\n--- Answer 1 (baseline Top 3) ---\")
    print(p1.answer)
    print(\"\\n--- Answer 2 (re-ranked Top 3) ---\")
    print(p2.answer)
"""
)

md(
    """
## 9. Rank-shift table and qualitative comparison

For the report: record which chunk moved (for example rank 8 → rank 1) and whether Answer 1 hallucinated, omitted an exception, or stated the general rule as if it were absolute.
"""
)

code(
    """
rank_df = pd.DataFrame(rank_rows)
latency_df = pd.DataFrame(latency_rows)
print(\"Gold-chunk rank shifts (positive shift = promoted by the Cross-Encoder)\")
display(rank_df)

print(\"\\nLatency (milliseconds)\")
display(latency_df)

summary = (
    latency_df.groupby(\"pipeline\")[[\"retrieve_ms\", \"rerank_ms\", \"retrieval_total_ms\", \"generate_ms\"]]
    .mean()
    .round(2)
)
print(\"\\nMean latency by pipeline\")
display(summary)

rank_df.to_csv(OUT_DIR / \"rank_shifts.csv\", index=False)
latency_df.to_csv(OUT_DIR / \"latency.csv\", index=False)

fig, ax = plt.subplots(figsize=(7, 4))
means = latency_df.groupby(\"pipeline\")[\"retrieval_total_ms\"].mean()
ax.bar([\"Baseline retrieve\", \"Retrieve + re-rank\"], [means.get(\"baseline\", 0), means.get(\"rerank\", 0)], color=[\"#4C78A8\", \"#F58518\"])
ax.set_ylabel(\"Mean retrieval latency (ms)\")
ax.set_title(\"Bi-Encoder only vs Bi-Encoder + Cross-Encoder\")
fig.tight_layout()
fig.savefig(OUT_DIR / \"latency_retrieval.png\", dpi=140)
plt.show()

# Persist full run traces for the written report (answers + top chunks).
serializable = {}
for qid, pair in all_runs.items():
    serializable[qid] = {}
    for name, res in pair.items():
        serializable[qid][name] = {
            \"query\": res.query,
            \"answer\": res.answer,
            \"retrieve_ms\": res.retrieve_ms,
            \"rerank_ms\": res.rerank_ms,
            \"generate_ms\": res.generate_ms,
            \"top10\": [
                {
                    \"rank\": h[\"rank\"],
                    \"baseline_rank\": h.get(\"baseline_rank\"),
                    \"rule_id\": h.get(\"rule_id\"),
                    \"rule_ids\": h.get(\"rule_ids\"),
                    \"score\": h.get(\"score\"),
                    \"ce_score\": h.get(\"ce_score\"),
                    \"text\": h[\"text\"][:800],
                }
                for h in res.hits
            ],
        }

with open(OUT_DIR / \"run_trace.json\", \"w\", encoding=\"utf-8\") as f:
    json.dump(serializable, f, indent=2)
print(\"Wrote\", OUT_DIR / \"rank_shifts.csv\")
print(\"Wrote\", OUT_DIR / \"latency.csv\")
print(\"Wrote\", OUT_DIR / \"run_trace.json\")
"""
)

md(
    """
## 10. Sanity check that both pipelines share the same corpus and prompt

If this cell prints matching prompt prefixes and matching candidate-generator scores, the contrast is a fair test of re-ranking rather than a change in chunking or wording.
"""
)

code(
    """
# Retrieval-only check so we do not spend extra GPU time regenerating answers.
sample = EVAL_QUERIES[0][\"query\"]
base_hits = baseline_retrieve(sample, TOP_K_RETRIEVE)
ce_hits = rerank(sample, base_hits)
p1_prompt = build_prompt(sample, base_hits[:TOP_K_CONTEXT])
p2_prompt = build_prompt(sample, ce_hits[:TOP_K_CONTEXT])
print(\"Same prompt header?\", p1_prompt.split(\"Retrieved rules:\")[0] == p2_prompt.split(\"Retrieved rules:\")[0])
print(\"Baseline Top-10 rows:\", [h[\"row\"] for h in base_hits])
print(\"Re-rank still uses those rows:\", sorted(h[\"row\"] for h in ce_hits) == sorted(h[\"row\"] for h in base_hits))
print(\"Top-3 rule ids baseline:\", [h[\"rule_id\"] for h in base_hits[:3]])
print(\"Top-3 rule ids rerank:  \", [h[\"rule_id\"] for h in ce_hits[:3]])
"""
)

md(
    """
## 11. Notes for the written report (write this part in your own words)

The assignment forbids an AI-written report. Use the tables in `outputs/` and describe what **you** observed. Cover:

1. **Architecture comparison** — independent embeddings + cosine vs joint self-attention over `(query, chunk)`.
2. **Rank shifts** — for each failure query, baseline rank of the gold rule vs re-ranked rank, and how Answer 1 vs Answer 2 changed (hallucination, missing exception, polarity error).
3. **Latency** — copy the mean millisecond table. Re-ranking adds one MiniLM forward pass over 10 pairs; generation still dominates wall time.
4. **Scaling** — a Cross-Encoder costs a full transformer pass per pair. For ~N chunks that is O(N) expensive forwards per query. A Bi-Encoder builds N vectors once, then a query is one embed + cheap inner products (or ANN). That is why we retrieve 10 with the Bi-Encoder and only then re-rank.

Do not paste this notebook's markdown into the report unchanged. The markers want your observations from the actual ranks this run produced.
"""
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": cells,
}
out = Path(__file__).resolve().parent / "rag_baseline_vs_rerank.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("Wrote", out, "cells", len(cells))
