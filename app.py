"""
Wikipedia RAG Chatbot Backend - Clean Version
- No torch/sentence-transformers (RAM-friendly)
- BM25 retrieval (pure Python)
- Model rotation for 429 errors
- Q&A caching
"""

import os, json, time, hashlib, re, logging, math
from pathlib import Path
from typing import Optional
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from rapidfuzz import process, fuzz
from openai import OpenAI, RateLimitError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

HEADERS = {"User-Agent": "WikiRAGBot/1.0 (Educational) requests/2.x"}
RATE_LIMIT_DELAY = 0.8

# ── Free model rotation list ────────────────────────────────────
FREE_MODELS = [
    "openai/gpt-oss-20b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-3-27b-it:free",
    "qwen/qwen3-8b:free",
    "deepseek/deepseek-r1-0528:free",
]

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# ── In-memory store ─────────────────────────────────────────────
article_store: dict = {}

# ══════════════════════════════════════════════════════════════
#  BM25 RETRIEVAL — pure Python, zero RAM overhead
# ══════════════════════════════════════════════════════════════

def tokenize(text: str) -> list:
    return re.findall(r"[a-z0-9]+", text.lower())

def bm25_score(query_tokens: list, chunk: str, all_chunks: list) -> float:
    k1, b = 1.5, 0.75
    avgdl = sum(len(c.split()) for c in all_chunks) / max(len(all_chunks), 1)
    doc_tokens = tokenize(chunk)
    doc_len = len(doc_tokens)
    doc_freq = Counter(doc_tokens)
    N = len(all_chunks)
    score = 0.0
    for token in set(query_tokens):
        tf = doc_freq.get(token, 0)
        if tf == 0:
            continue
        df = sum(1 for c in all_chunks if token in tokenize(c))
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / max(avgdl, 1)))
        score += idf * tf_norm
    return score

def retrieve_chunks_bm25(chunks: list, question: str, top_k: int = 12) -> list:
    query_tokens = tokenize(question)
    # Expand query: also search for any bare numbers/years mentioned in the question
    numbers = re.findall(r'\b(\d{4}|\d{1,3})\b', question)
    for n in numbers:
        if n not in query_tokens:
            query_tokens.append(n)
    scored = [(bm25_score(query_tokens, c, chunks), c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [c for _, c in scored[:top_k] if _ > 0]
    top = results if results else chunks[:top_k]
    # ── Debug: log the top retrieved chunks ────────────────────────
    logger.info(f"[BM25] Retrieved {len(top)} chunks for query: '{question}'")
    for i, (score, chunk) in enumerate(scored[:min(3, len(scored))]):
        preview = chunk[:120].replace("\n", " ")
        logger.info(f"  [{i+1}] score={score:.3f} | {preview}")
    return top

# ── Cache helpers ───────────────────────────────────────────────

def _ck(text: str) -> str:
    return hashlib.md5(text.lower().strip().encode()).hexdigest()

def cache_get(key: str) -> Optional[dict]:
    p = CACHE_DIR / f"{key}.json"
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return None
    return None

def cache_set(key: str, data: dict):
    p = CACHE_DIR / f"{key}.json"
    with open(p, "w") as f:
        json.dump(data, f, ensure_ascii=False)


def _make_absolute_url(src: str) -> Optional[str]:
    if not src:
        return None
    src = src.strip()
    if src.startswith('//'):
        return 'https:' + src
    if src.startswith('/'):
        return 'https://en.wikipedia.org' + src
    if src.startswith('http://') or src.startswith('https://'):
        return src
    return None

# ── Wikipedia helpers ───────────────────────────────────────────

def wikipedia_search(query: str) -> list:
    """OpenSearch autocomplete — fast but requires near-exact spelling."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {"action": "opensearch", "search": query,
               "limit": 10, "namespace": 0, "format": "json"}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data[1] if len(data) > 1 else []
    except Exception as e:
        logger.error(f"WP opensearch error: {e}")
        return []

def wikipedia_fulltext_search(query: str) -> list:
    """Full-text search with typo tolerance — handles misspellings."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 10,
        "srnamespace": 0,
        "srqiprofile": "classic_noboostlinks",  # handles fuzzy/typos
        "format": "json",
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        results = data.get("query", {}).get("search", [])
        return [item["title"] for item in results]
    except Exception as e:
        logger.error(f"WP fulltext search error: {e}")
        return []

def resolve_title(query: str) -> Optional[str]:
    ck = _ck(f"title_{query}")
    cached = cache_get(ck)
    if cached:
        return cached.get("title")

    # 1. Try OpenSearch (fast, autocomplete)
    candidates = wikipedia_search(query)

    # 2. Try with underscores
    if not candidates:
        candidates = wikipedia_search(query.replace(" ", "_"))

    # 3. Fall back to full-text search (typo-tolerant)
    if not candidates:
        candidates = wikipedia_fulltext_search(query)

    if not candidates:
        return None

    # Pick best fuzzy match, but fall back to top result if no strong match
    best = process.extractOne(query, candidates, scorer=fuzz.WRatio)
    title = best[0] if (best and best[1] >= 40) else candidates[0]
    cache_set(ck, {"title": title})
    time.sleep(RATE_LIMIT_DELAY)
    return title

def _get_section_heading(element) -> str:
    """Walk backwards through siblings/ancestors to find the nearest heading."""
    # Check preceding siblings first
    for sibling in element.find_previous_siblings():
        tag = getattr(sibling, "name", None)
        if tag in ("h1", "h2", "h3", "h4"):
            return sibling.get_text(" ", strip=True)
    # Fall back to parent's preceding siblings
    parent = element.parent
    if parent:
        for sibling in parent.find_previous_siblings():
            tag = getattr(sibling, "name", None)
            if tag in ("h1", "h2", "h3", "h4"):
                return sibling.get_text(" ", strip=True)
    return "Table"


def extract_table_chunks(content_div) -> list:
    """
    Extract all content tables from the Wikipedia article and convert each
    into a structured text chunk for embedding/indexing.

    Output format per table:
        <Section Heading>:
        Row Label | col1: val1 | col2: val2 | ...
        Row Label | col1: val1 | col2: val2 | ...
    """
    table_chunks = []
    # Skip known non-content table classes
    skip_classes = {"infobox", "navbox", "ambox", "wikitable-stub",
                    "mbox-small", "toc", "sidebar"}

    tables = content_div.find_all("table")
    logger.info(f"[TABLE] Found {len(tables)} raw tables in article")

    for table in tables:
        classes = set(table.get("class", []))
        if classes & skip_classes:
            continue

        rows = table.find_all("tr")
        if not rows:
            continue

        # ── Extract column headers — handle multi-row <th> headers ──
        # Collect all header rows; the LAST one has the most granular labels
        header_rows = [r for r in rows if r.find("th")]
        header_row = header_rows[-1] if header_rows else None

        if header_row:
            headers = [th.get_text(" ", strip=True) for th in header_row.find_all("th")]
            headers = [re.sub(r"\s+", " ", h).strip() for h in headers]
            # If multi-row headers exist, pull a human-readable label from row 0
            if len(header_rows) > 1:
                first_ths = header_rows[0].find_all("th")
                if first_ths:
                    label_candidate = first_ths[0].get_text(" ", strip=True)
                    label_candidate = re.sub(r"\s+", " ", label_candidate).strip()
                    # Store as extra context; will be prepended to chunk
                    table_caption = label_candidate
                else:
                    table_caption = None
            else:
                table_caption = None
        else:
            # No <th> headers — try first row as plain text labels
            first_cells = rows[0].find_all(["td", "th"])
            headers = [c.get_text(" ", strip=True) for c in first_cells]
            headers = [re.sub(r"\s+", " ", h).strip() for h in headers]
            table_caption = None

        if len(headers) < 2:
            # Single-column or empty table — not useful
            continue

        # ── Determine section label ─────────────────────────────────
        section_label = _get_section_heading(table)
        # If we found a table-level caption (e.g. "Atmospheric pressure comparison"), use it
        if table_caption:
            section_label = table_caption
        # All headers from the last header row are the column names
        # (they align with data cells after the row-label cell)
        value_cols = headers  # full list: e.g. ["kilopascal", "psi"]

        # ── Build one line per data row ─────────────────────────────
        lines = []
        header_row_set = set(id(r) for r in header_rows)  # exclude ALL header rows
        data_rows = [r for r in rows if id(r) not in header_row_set]
        for row in data_rows:
            cells = row.find_all(["td", "th"])
            if not cells:
                continue
            cell_texts = [re.sub(r"\s+", " ", c.get_text(" ", strip=True)) for c in cells]
            # remove any bracketed reference markers like [12], [ 12 ], [156][181], etc.
            cell_texts = [re.sub(r"\[[^\]]*\]", "", t).strip() for t in cell_texts]

            if not cell_texts or not cell_texts[0]:   # skip blank rows
                continue

            # Align headers with cell values. Some tables use a separate row-label
            # (e.g. first cell is the row name) while others have headers for every
            # column. Handle both cases:
            parts = []
            if len(cell_texts) == len(value_cols):
                # full alignment: header_i -> cell_i
                parts = [f"{h}: {v}" for h, v in zip(value_cols, cell_texts) if h and v]
            elif len(cell_texts) == len(value_cols) + 1:
                # first cell is a row label, remaining map to headers
                row_label = cell_texts[0]
                parts = [row_label] + [f"{h}: {v}" for h, v in zip(value_cols, cell_texts[1:]) if h and v]
            else:
                # fallback: join available cells
                parts = [c for c in cell_texts if c]

            if len(parts) < 1:
                continue
            lines.append(" | ".join(parts))

        if not lines:
            continue

        chunk = f"{section_label}:\n" + "\n".join(lines)
        table_chunks.append(chunk)
        logger.info(f"[TABLE] Chunk created — '{section_label}' ({len(lines)} rows)")

    logger.info(f"[TABLE] Total table chunks created: {len(table_chunks)}")
    return table_chunks


def scrape_wikipedia(title: str) -> dict:
    # Version tag — bump this to invalidate caches when extraction logic changes
    CACHE_VERSION = "v2"
    ck = _ck(f"scrape_{CACHE_VERSION}_{title}")
    cached = cache_get(ck)
    if cached:
        logger.info(f"Cache hit: {title}")
        return cached

    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        time.sleep(RATE_LIMIT_DELAY)
    except Exception as e:
        return {"error": str(e)}

    soup = BeautifulSoup(resp.text, "html.parser")
    table_count = len(soup.find_all("table"))
    logger.info(f"[SCRAPE] Total tables on page (before noise removal): {table_count}")

    # Infobox
    infobox_data = {}
    image_url = None
    infobox = soup.find("table", class_=re.compile(r"infobox"))
    if infobox:
        for row in infobox.find_all("tr"):
            th, td = row.find("th"), row.find("td")
            if th and td:
                k = th.get_text(strip=True)
                v = td.get_text(" ", strip=True)
                if k and v:
                    infobox_data[k] = v
        img = infobox.find("img")
        if img:
            image_url = _make_absolute_url(img.get("src") or img.get("data-src"))

    content = soup.find("div", id="mw-content-text")
    if not content:
        return {"error": "Could not find main content div."}

    # ── Extract table chunks BEFORE removing noise elements ─────────
    table_chunks = extract_table_chunks(content)

    # Remove noise (after table extraction so we don't lose content tables)
    for sel in ["sup", "div.navbox", "div.reflist", "table.navbox", "table.ambox",
                "div.hatnote", "div.toc", "div.mw-references-wrap", "div.thumb",
                "span.noprint", "div.noprint", "div.mw-editsection"]:
        for el in soup.select(sel):
            el.decompose()

    paragraphs = []
    if content:
        for p in content.find_all("p"):
            text = p.get_text(" ", strip=True)
            if len(text) > 60:
                text = re.sub(r"\[\d+\]", "", text)
                text = re.sub(r"\s+", " ", text).strip()
                paragraphs.append(text)

    result = {
        "title": title, "url": url,
        "paragraphs": paragraphs,
        "table_chunks": table_chunks,
        "table_count": table_count,
        "infobox": infobox_data,
        "image_url": image_url,
    }
    cache_set(ck, result)
    return result

def chunk_text(paragraphs: list, max_words: int = 300) -> list:
    chunks, cur, cur_len = [], [], 0
    for para in paragraphs:
        words = para.split()
        if cur_len + len(words) > max_words and cur:
            chunks.append(" ".join(cur))
            cur = cur[-30:]
            cur_len = len(cur)
        cur.extend(words)
        cur_len += len(words)
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def build_index(title: str, paragraphs: list, table_chunks: list = None):
    if title in article_store:
        return
    CACHE_VERSION = "v2"
    ck = _ck(f"chunks_{CACHE_VERSION}_{title}")
    cached = cache_get(ck)
    if cached:
        article_store[title] = {"chunks": cached["chunks"]}
        logger.info(f"Loaded chunks from cache: {title} ({len(cached['chunks'])} total chunks)")
        return
    para_chunks = chunk_text(paragraphs)
    tbl_chunks = table_chunks or []
    all_chunks = para_chunks + tbl_chunks
    article_store[title] = {"chunks": all_chunks}
    cache_set(ck, {"chunks": all_chunks})
    logger.info(
        f"[INDEX] '{title}': {len(para_chunks)} paragraph chunks + "
        f"{len(tbl_chunks)} table chunks = {len(all_chunks)} total"
    )

# ── LLM with model rotation ─────────────────────────────────────

def _strip_markdown(text: str) -> str:
    """Remove markdown formatting that models add despite instructions."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)           # **bold**
    text = re.sub(r'__(.+?)__', r'\1', text)               # __bold__
    text = re.sub(r'\*(.+?)\*', r'\1', text)               # *italic*
    text = re.sub(r'_(.+?)_', r'\1', text)                 # _italic_
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # # headers
    text = re.sub(r'^[-•]\s+', '• ', text, flags=re.MULTILINE)  # bullets
    text = re.sub(r'\n{3,}', '\n\n', text)                 # excess blank lines
    return text.strip()


def _try_llm_call(model: str, messages: list, max_tokens: int = 600) -> Optional[str]:
    """Attempt a single LLM call. Returns response text or None on failure."""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=max_tokens,
        timeout=45,
    )
    return resp.choices[0].message.content.strip()


def rewrite_question(question: str, history: list) -> str:
    """
    If the question is a short follow-up (like "in kilopascal?"),
    use the conversation history to rewrite it as a fully self-contained question.
    Returns the rewritten question, or the original if no rewrite is needed.
    """
    # Only attempt rewrite for short / pronoun-heavy questions
    words = question.split()
    has_pronoun = any(w.lower() in ('it', 'its', 'that', 'this', 'they', 'them',
                                     'those', 'same', 'there', 'he', 'she',
                                     'what', 'which') for w in words)
    is_short = len(words) <= 7
    starts_with_preposition = words[0].lower() in ('in', 'at', 'by', 'to', 'of',
                                                     'for', 'from', 'with', 'about',
                                                     'what', 'how', 'why', 'when',
                                                     'where', 'and', 'but')

    if not history or (not has_pronoun and not is_short and not starts_with_preposition):
        return question  # standalone question, no rewrite needed

    # Build a compact history string (last 6 turns max)
    history_text = ""
    for turn in history[-6:]:
        role = "User" if turn["role"] == "user" else "Assistant"
        history_text += f"{role}: {turn['content']}\n"

    rewrite_prompt = (
        f"Given this conversation:\n{history_text}\n"
        f"The user now asks: \"{question}\"\n"
        "Rewrite this as a single, fully self-contained question that includes all "
        "necessary context from the conversation. "
        "Return ONLY the rewritten question, nothing else."
    )

    for model in FREE_MODELS:
        try:
            rewritten = _try_llm_call(
                model,
                [{"role": "user", "content": rewrite_prompt}],
                max_tokens=80
            )
            if rewritten:
                rewritten = rewritten.strip().strip('"').strip("'")
                logger.info(f"[REWRITE] '{question}' → '{rewritten}'")
                return rewritten
        except RateLimitError:
            continue
        except Exception as e:
            logger.warning(f"Rewrite failed on {model}: {e}")
            continue

    return question  # fallback: use original


def answer_with_llm(question: str, chunks: list, title: str, infobox: dict,
                    history: list = None) -> str:
    # Only cache standalone questions (no history context)
    qa_key = _ck(f"qa_{title}_{question}") if not history else None
    if qa_key:
        cached = cache_get(qa_key)
        if cached and cached.get("answer"):
            logger.info("QA cache hit")
            return cached["answer"]

    context = "\n\n---\n\n".join(chunks)
    ib = ""
    if infobox:
        lines = [f"{k}: {v}" for k, v in list(infobox.items())[:12]]
        ib = "KEY FACTS:\n" + "\n".join(lines) + "\n\n"

    system_msg = (
        "You are a factual assistant. You answer questions ONLY using the Wikipedia content provided. "
        "CRITICAL FORMATTING RULES — violating these is not allowed:\n"
        "  1. Use PLAIN TEXT only. No markdown whatsoever.\n"
        "  2. Do NOT write **bold**, *italic*, __underline__, or `code`.\n"
        "  3. Do NOT write # headers or ## subheadings.\n"
        "  4. Do NOT use - or * as bullet points. Use plain numbered lists (1. 2. 3.) if needed.\n"
        "  5. If the answer is not in the provided context, reply exactly: "
        '"Information not found in the Wikipedia article."\n'
        "  6. Be concise and direct."
    )

    context_block = f"{ib}WIKIPEDIA CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER (plain text only, no markdown):"

    # Build messages: system + optional prior turns + current question
    messages = [{"role": "system", "content": system_msg}]

    if history:
        # Inject prior conversation turns (last 6 turns) before the context block
        for turn in history[-6:]:
            messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": context_block})

    for i, model in enumerate(FREE_MODELS):
        if i > 0:
            wait = min(2 ** i, 12)
            logger.info(f"Waiting {wait}s before trying {model}…")
            time.sleep(wait)
        try:
            logger.info(f"Trying: {model}")
            answer = _try_llm_call(model, messages)
            answer = _strip_markdown(answer)
            if qa_key:
                cache_set(qa_key, {"answer": answer, "model": model})
            logger.info(f"✓ Answer from {model}")
            return answer
        except RateLimitError:
            logger.warning(f"429 on {model} — rotating")
            continue
        except Exception as e:
            logger.error(f"Error on {model}: {e}")
            continue

    return (
        "All free AI models are currently rate-limited. "
        "Please wait 30 seconds and try again."
    )

# ── Routes ──────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/search", methods=["POST"])
def search():
    query = (request.json or {}).get("query", "").strip()
    if not query:
        return jsonify({"error": "Query is required"}), 400
    title = resolve_title(query)
    if not title:
        return jsonify({"error": f"No Wikipedia article found for '{query}'"}), 404
    article = scrape_wikipedia(title)
    if "error" in article:
        return jsonify({"error": article["error"]}), 500
    build_index(title, article["paragraphs"], article.get("table_chunks", []))
    return jsonify({
        "title": title,
        "url": article["url"],
        "table_count": article["table_count"],
        "table_chunk_count": len(article.get("table_chunks", [])),
        "chunk_count": len(article_store.get(title, {}).get("chunks", [])),
        "cached": False,
        "summary": article["paragraphs"][0] if article["paragraphs"] else "",
        "infobox": dict(list(article.get("infobox", {}).items())[:8]),
        "image_url": article.get("image_url"),
    })

@app.route("/api/ask", methods=["POST"])
def chat():
    data = request.json or {}
    query = (data.get("query") or data.get("title") or "").strip()
    question = (data.get("question") or "").strip()
    history = data.get("history") or []  # list of {role, content} dicts
    if not query or not question:
        return jsonify({"error": "query and question are required"}), 400

    logger.info(f"Q: {question} | Topic: {query} | History turns: {len(history)}")
    title = resolve_title(query)
    if not title:
        return jsonify({"error": f"No Wikipedia article found for '{query}'"}), 404

    article = scrape_wikipedia(title)
    if "error" in article:
        return jsonify({"error": article["error"]}), 500

    build_index(title, article["paragraphs"], article.get("table_chunks", []))

    all_chunks = article_store.get(title, {}).get("chunks", [])
    if not all_chunks:
        return jsonify({"error": "No content found."}), 404

    # Rewrite ambiguous follow-up questions using conversation history
    resolved_question = rewrite_question(question, history) if history else question

    top_chunks = retrieve_chunks_bm25(all_chunks, resolved_question, top_k=12)
    answer = answer_with_llm(
        resolved_question, top_chunks, title,
        article.get("infobox", {}),
        history=history
    )

    return jsonify({
        "query": query,
        "resolved_title": title,
        "resolved_question": resolved_question,
        "answer": answer,
        "table_count": article["table_count"],
        "table_chunk_count": len(article.get("table_chunks", [])),
        "chunks_used": len(top_chunks),
        "article_url": article["url"],
        "cached": False,
    })

@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)