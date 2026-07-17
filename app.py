"""
Wikipedia RAG Chatbot Backend - v2
- Hybrid retrieval: BM25 + vector embeddings (all-MiniLM-L6-v2) + RRF fusion
- Cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2)
- LLM query expansion for retrieval
- Model rotation for 429 errors
- Q&A caching
"""

import os, json, time, hashlib, re, logging, math, warnings
from pathlib import Path
from typing import Optional
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

import requests
import numpy as np
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from rapidfuzz import process, fuzz
from openai import OpenAI, RateLimitError
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

HEADERS = {"User-Agent": "DeepQuery/2.0 (https://github.com/keshuuu800/Deepquery; keshavgupta1511@gmail.com) python-requests/2.x"}
RATE_LIMIT_DELAY = 0.8

# ── Groq primary models (fast, free tier) ───────────────────────
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "gemma2-9b-it",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
]

# ── OpenRouter fallback — auto-selects best available free model ──
OPENROUTER_MODELS = [
    "openrouter/free",   # Auto-routes to best free model available right now
]

# Keep FREE_MODELS as alias for any legacy references
FREE_MODELS = GROQ_MODELS

# Groq client (primary)
groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# OpenRouter client (fallback)
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# ── Embedding model for dense retrieval ─────────────────────────
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
RRF_K = 60  # constant for Reciprocal Rank Fusion

# ── In-memory store ─────────────────────────────────────────────
# {title: {"chunks": [...], "embeddings": np.array(...)}}
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

# ══════════════════════════════════════════════════════════════
#  DENSE RETRIEVAL — vector embeddings (semantic search)
# ══════════════════════════════════════════════════════════════

def compute_embeddings(chunks: list) -> np.ndarray:
    return EMBEDDING_MODEL.encode(chunks, show_progress_bar=False, normalize_embeddings=True)

def retrieve_chunks_vector(query: str, embeddings: np.ndarray, chunks: list, top_k: int = 12) -> list:
    query_emb = EMBEDDING_MODEL.encode([query], show_progress_bar=False, normalize_embeddings=True)
    sims = np.dot(embeddings, query_emb.T).flatten()
    top_indices = np.argsort(sims)[-top_k:][::-1]
    return [chunks[i] for i in top_indices if sims[i] > 0]

def _augment_queries(question: str, title: str) -> list:
    """Generate multiple search queries to maximize recall."""
    queries = [question]
    q_lower = question.lower().strip("?. ")
    if title:
        queries.append(f"{title}: {question}")
        is_definition_q = any(q_lower.startswith(p) for p in [
            "what is", "what are", "who is", "who was", "who are",
            "tell me about", "describe", "explain",
        ])
        if is_definition_q:
            queries.append(title)
            queries.append(f"{title} is")
        elif any(q_lower.startswith(p) for p in ["when was", "when did", "when is", "what year"]):
            queries.append(f"{title} date year founded born")
        elif any(w in q_lower for w in ["how many", "how much", "how tall", "how high",
                                         "height", "elevation", "altitude", "population",
                                         "score", "runs", "average", "record"]):
            queries.append(f"{title} statistics numbers data")
        elif any(q_lower.startswith(p) for p in ["why", "what caused", "reason"]):
            queries.append(f"{title} cause reason why")
        elif any(q_lower.startswith(p) for p in ["where", "location", "situated"]):
            queries.append(f"{title} location situated country")
    unique = []
    for q in queries:
        if q not in unique:
            unique.append(q)
    return unique

def hybrid_retrieve(chunks: list, embeddings: np.ndarray, question: str, title: str = "", top_k: int = 12) -> list:
    queries = _augment_queries(question, title)
    combined_scores = {}
    
    # Run retrieval for all augmented queries and merge results by best RRF score
    for query in queries:
        bm25_results = retrieve_chunks_bm25(chunks, query, top_k=top_k * 2)
        vec_results = retrieve_chunks_vector(query, embeddings, chunks, top_k=top_k * 2)
        
        for rank, chunk in enumerate(bm25_results):
            combined_scores[chunk] = combined_scores.get(chunk, 0) + 1 / (RRF_K + rank + 1)
        for rank, chunk in enumerate(vec_results):
            combined_scores[chunk] = combined_scores.get(chunk, 0) + 1 / (RRF_K + rank + 1)

    reranked = sorted(combined_scores, key=lambda c: combined_scores[c], reverse=True)
    
    # Guarantee intro chunks are included for definition queries
    intro_anchors = chunks[:min(3, len(chunks))]
    for chunk in intro_anchors:
        if chunk not in reranked:
            reranked.insert(0, chunk) # Place at top or guarantee presence
            
    logger.info(f"[HYBRID] Multi-query search merged into top {top_k}")
    return reranked[:top_k]

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

def wikipedia_search_rest(query: str) -> list:
    """Use Wikipedia REST API — works from datacenter IPs, no rate-limit block."""
    try:
        url = f"https://en.wikipedia.org/w/rest.php/v1/search/title"
        r = requests.get(url, params={"q": query, "limit": 10},
                         headers=HEADERS, timeout=10)
        if r.status_code == 200:
            data = r.json()
            return [p["title"] for p in data.get("pages", [])]
    except Exception as e:
        logger.warning(f"WP REST search error: {e}")
    return []

def wikipedia_search(query: str) -> list:
    """OpenSearch autocomplete via w/api.php."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {"action": "opensearch", "search": query,
               "limit": 10, "namespace": 0, "format": "json"}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            logger.warning(f"WP opensearch HTTP {r.status_code} for '{query}'")
            return []
        data = r.json()
        return data[1] if len(data) > 1 else []
    except Exception as e:
        logger.error(f"WP opensearch error: {e}")
        return []

def wikipedia_fulltext_search(query: str) -> list:
    """Full-text search with typo tolerance via w/api.php."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 10,
        "srnamespace": 0,
        "format": "json",
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            logger.warning(f"WP fulltext HTTP {r.status_code} for '{query}'")
            return []
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

    # 1. REST API — most reliable from datacenter IPs
    candidates = wikipedia_search_rest(query)

    # 2. OpenSearch autocomplete
    if not candidates:
        candidates = wikipedia_search(query)

    # 3. Try with underscores
    if not candidates:
        candidates = wikipedia_search(query.replace(" ", "_"))

    # 4. Full-text search fallback
    if not candidates:
        candidates = wikipedia_fulltext_search(query)

    if not candidates:
        logger.error(f"resolve_title: no candidates found for '{query}'")
        return None

    # Pick best fuzzy match, fall back to top result
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


def fetch_explaintext(title: str) -> str:
    """Fetch clean wikitext extract plain text from Wikipedia."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "titles": title,
        "format": "json",
        "redirects": 1,
        "explaintext": 1,
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            pages = r.json().get("query", {}).get("pages", {})
            for page in pages.values():
                return page.get("extract", "")
    except Exception as e:
        logger.error(f"Explaintext fetch error: {e}")
    return ""


def scrape_wikipedia(title: str) -> dict:
    # Version tag — bump this to invalidate caches when extraction logic changes
    CACHE_VERSION = "v4"
    ck = _ck(f"scrape_{CACHE_VERSION}_{title}")
    cached = cache_get(ck)
    if cached:
        logger.info(f"Cache hit: {title}")
        return cached

    # Fetch clean wikitext extracts first to avoid HTML noise in prose RAG
    explaintext = fetch_explaintext(title)

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

    # Extract in-article images with their nearest section heading and alt text
    images = []
    def _px_size_from_url(u: str) -> int:
        m = re.search(r"/(\d+)px-", u)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return 0
        m2 = re.search(r"(\d+)px-", u)
        if m2:
            try:
                return int(m2.group(1))
            except Exception:
                return 0
        return 0

    def _is_decorative(u: str, alt: str) -> bool:
        lower = (u or "").lower()
        # common tiny icon patterns from Wikipedia/Commons
        if any(k in lower for k in ("transparent", "nuvola", "maki-", "20px", "40px", "maps.wikimedia.org/img/osm", "thumb/20px", "thumb/40px")):
            return True
        # exclude very small thumbnails (px size embedded in URL)
        px = _px_size_from_url(lower)
        if px and px < 80:
            return True
        # exclude data URIs or empty
        if lower.startswith('data:'):
            return True
        return False

    for img in content.find_all('img'):
        src = img.get('src') or img.get('data-src') or ''
        url_abs = _make_absolute_url(src)
        if not url_abs:
            continue
        alt = img.get('alt') or ''
        section = _get_section_heading(img)
        if _is_decorative(url_abs, alt):
            continue
        images.append({"url": url_abs, "alt": alt, "section": section})

    # Process paragraphs: prioritize clean explaintext if available
    paragraphs = []
    if explaintext:
        # Split explaintext into clean paragraphs, removing headers like "== History =="
        for line in explaintext.split("\n"):
            line = line.strip()
            if not line or (line.startswith("==") and line.endswith("==")):
                continue
            if len(line) > 60:
                paragraphs.append(line)

    # If explaintext was empty, fallback to soup parsing
    if not paragraphs:
        # Remove noise (after table extraction so we don't lose content tables)
        for sel in ["sup", "div.navbox", "div.reflist", "table.navbox", "table.ambox",
                    "div.hatnote", "div.toc", "div.mw-references-wrap", "div.thumb",
                    "span.noprint", "div.noprint", "div.mw-editsection"]:
            for el in soup.select(sel):
                el.decompose()

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
        "images": images,
    }
    cache_set(ck, result)
    return result

def chunk_text(paragraphs: list, max_words: int = 300, overlap: int = 50) -> list:
    chunks, cur, cur_len = [], [], 0
    for para in paragraphs:
        words = para.split()
        while cur_len + len(words) > max_words and cur:
            split_at = max_words - overlap
            chunks.append(" ".join(cur[:split_at]))
            cur = cur[split_at:]
            cur_len = len(cur)
        cur.extend(words)
        cur_len += len(words)
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def build_index(title: str, paragraphs: list, table_chunks: list = None):
    CACHE_VERSION = "v3"
    ck = _ck(f"chunks_{CACHE_VERSION}_{title}")
    cached = cache_get(ck)
    if cached:
        chunks = cached["chunks"]
        embeddings = np.array(cached["embeddings"]) if "embeddings" in cached else compute_embeddings(chunks)
        article_store[title] = {"chunks": chunks, "embeddings": embeddings}
        logger.info(f"Loaded from cache: {title} ({len(chunks)} chunks)")
        return

    para_chunks = chunk_text(paragraphs)
    tbl_chunks = table_chunks or []
    all_chunks = para_chunks + tbl_chunks
    embeddings = compute_embeddings(all_chunks)
    article_store[title] = {"chunks": all_chunks, "embeddings": embeddings}
    cache_set(ck, {"chunks": all_chunks, "embeddings": embeddings.tolist()})
    logger.info(
        f"[INDEX] '{title}': {len(para_chunks)} paragraph + "
        f"{len(tbl_chunks)} table = {len(all_chunks)} chunks"
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


def _try_llm_call(model: str, messages: list, max_tokens: int = 600,
                  llm_client=None) -> Optional[str]:
    """Attempt a single LLM call. Uses groq_client by default."""
    c = llm_client or groq_client
    resp = c.chat.completions.create(
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

    # Try Groq first, then OpenRouter on rate-limit
    for model, llm_client in [(m, groq_client) for m in GROQ_MODELS] + [(m, client) for m in OPENROUTER_MODELS]:
        try:
            rewritten = _try_llm_call(
                model,
                [{"role": "user", "content": rewrite_prompt}],
                max_tokens=80, llm_client=llm_client
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

    # ── Step 1: Try all Groq models (primary) ───────────────────
    for i, model in enumerate(GROQ_MODELS):
        if i > 0:
            time.sleep(min(2 ** i, 8))
        try:
            logger.info(f"[GROQ] Trying: {model}")
            answer = _try_llm_call(model, messages, llm_client=groq_client)
            answer = _strip_markdown(answer)
            if qa_key:
                cache_set(qa_key, {"answer": answer, "model": f"groq/{model}"})
            logger.info(f"✓ Answer from groq/{model}")
            return answer
        except RateLimitError:
            logger.warning(f"[GROQ] 429 on {model} — rotating")
            continue
        except Exception as e:
            logger.error(f"[GROQ] Error on {model}: {e}")
            continue

    # ── Step 2: Groq exhausted — fall back to OpenRouter :free ───
    logger.info("[OPENROUTER] All Groq models rate-limited, switching to OpenRouter...")
    for i, model in enumerate(OPENROUTER_MODELS):
        if i > 0:
            time.sleep(min(2 ** i, 12))
        try:
            logger.info(f"[OPENROUTER] Trying: {model}")
            answer = _try_llm_call(model, messages, llm_client=client)
            answer = _strip_markdown(answer)
            if qa_key:
                cache_set(qa_key, {"answer": answer, "model": model})
            logger.info(f"✓ Answer from openrouter/{model}")
            return answer
        except RateLimitError:
            logger.warning(f"[OPENROUTER] 429 on {model} — rotating")
            continue
        except Exception as e:
            logger.error(f"[OPENROUTER] Error on {model}: {e}")
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

    entry = article_store.get(title, {})
    all_chunks = entry.get("chunks", [])
    embeddings = entry.get("embeddings")
    if not all_chunks:
        return jsonify({"error": "No content found."}), 404

    # Rewrite ambiguous follow-up questions using conversation history
    resolved_question = rewrite_question(question, history) if history else question

    # Fast hybrid retrieval: BM25 + vector embeddings (with query expansion, anchors)
    if embeddings is not None:
        top_chunks = hybrid_retrieve(all_chunks, embeddings, resolved_question, title=title, top_k=10)
    else:
        top_chunks = retrieve_chunks_bm25(all_chunks, resolved_question, top_k=10)

    answer = answer_with_llm(
        resolved_question, top_chunks, title,
        article.get("infobox", {}),
        history=history
    )

    # Semantic context-based image matching using local sentence-transformer model
    imgs = article.get("images") or []
    scored_images = []
    
    # Pre-embed resolved question
    try:
        q_emb = EMBEDDING_MODEL.encode([resolved_question], show_progress_bar=False, normalize_embeddings=True)[0]
    except Exception as e:
        logger.error(f"Image query embedding error: {e}")
        q_emb = None

    if q_emb is not None:
        for im in imgs:
            url_val = im.get("url") or ""
            if not url_val:
                continue
            fname = url_val.split('/')[-1].replace('_', ' ')
            # Combine alt text, section heading, and filename for semantic matching
            candidate_text = f"{im.get('alt','')} {im.get('section','')} {fname}".strip()
            
            try:
                # Embed the candidate image description
                img_emb = EMBEDDING_MODEL.encode([candidate_text], show_progress_bar=False, normalize_embeddings=True)[0]
                # Cosine similarity (normalized inner product)
                score = float(np.dot(q_emb, img_emb))
            except Exception as e:
                logger.error(f"Image candidate embedding error: {e}")
                score = 0.0
                
            scored_images.append({
                "url": url_val,
                "alt": im.get('alt',''),
                "section": im.get('section',''),
                "score": round(score, 3)
            })
            
        # Sort images by semantic similarity score descending
        scored_images.sort(key=lambda x: x['score'], reverse=True)

    # Threshold cutoff (0.28 similarity required for matching context)
    top_image = None
    if scored_images and scored_images[0]['score'] >= 0.28:
        top_image = scored_images[0]['url']
        logger.info(f"[IMAGE] Found semantic match: {top_image} with score {scored_images[0]['score']}")
    else:
        # Strict context matching: do NOT fall back to general infobox image if she is not mentioned in context
        logger.info("[IMAGE] No highly relevant context image found. Rendering text only.")

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
        "image": top_image,
        "images": scored_images,
    })

@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)