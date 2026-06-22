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
    return results if results else chunks[:top_k]

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

# ── Wikipedia helpers ───────────────────────────────────────────

def wikipedia_search(query: str) -> list:
    url = "https://en.wikipedia.org/w/api.php"
    params = {"action": "opensearch", "search": query,
               "limit": 10, "namespace": 0, "format": "json"}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data[1] if len(data) > 1 else []
    except Exception as e:
        logger.error(f"WP search error: {e}")
        return []

def resolve_title(query: str) -> Optional[str]:
    ck = _ck(f"title_{query}")
    cached = cache_get(ck)
    if cached:
        return cached.get("title")
    candidates = wikipedia_search(query)
    if not candidates:
        candidates = wikipedia_search(query.replace(" ", "_"))
    if not candidates:
        return None
    best = process.extractOne(query, candidates, scorer=fuzz.WRatio)
    title = best[0] if (best and best[1] >= 60) else candidates[0]
    cache_set(ck, {"title": title})
    time.sleep(RATE_LIMIT_DELAY)
    return title

def scrape_wikipedia(title: str) -> dict:
    ck = _ck(f"scrape_{title}")
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

    # Infobox
    infobox_data = {}
    infobox = soup.find("table", class_=re.compile(r"infobox"))
    if infobox:
        for row in infobox.find_all("tr"):
            th, td = row.find("th"), row.find("td")
            if th and td:
                k = th.get_text(strip=True)
                v = td.get_text(" ", strip=True)
                if k and v:
                    infobox_data[k] = v

    # Remove noise
    for sel in ["sup", "div.navbox", "div.reflist", "table.navbox", "table.ambox",
                "div.hatnote", "div.toc", "div.mw-references-wrap", "div.thumb",
                "span.noprint", "div.noprint", "div.mw-editsection"]:
        for el in soup.select(sel):
            el.decompose()

    paragraphs = []
    content = soup.find("div", id="mw-content-text")
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
        "table_count": table_count,
        "infobox": infobox_data,
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

def build_index(title: str, paragraphs: list):
    if title in article_store:
        return
    ck = _ck(f"chunks_{title}")
    cached = cache_get(ck)
    if cached:
        article_store[title] = {"chunks": cached["chunks"]}
        logger.info(f"Loaded chunks from cache: {title}")
        return
    chunks = chunk_text(paragraphs)
    article_store[title] = {"chunks": chunks}
    cache_set(ck, {"chunks": chunks})
    logger.info(f"Indexed {len(chunks)} chunks for: {title}")

# ── LLM with model rotation ─────────────────────────────────────

def answer_with_llm(question: str, chunks: list, title: str, infobox: dict) -> str:
    qa_key = _ck(f"qa_{title}_{question}")
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

    user_msg = f"""{ib}WIKIPEDIA CONTEXT:
{context}

QUESTION: {question}

ANSWER (plain text only, no markdown):"""

    for i, model in enumerate(FREE_MODELS):
        if i > 0:
            wait = min(2 ** i, 12)
            logger.info(f"Waiting {wait}s before trying {model}…")
            time.sleep(wait)
        try:
            logger.info(f"Trying: {model}")
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=600,
                timeout=45,
            )
            answer = resp.choices[0].message.content.strip()
            # Strip markdown that models add despite being told not to
            answer = re.sub(r'\*\*(.+?)\*\*', r'\1', answer)           # **bold**
            answer = re.sub(r'__(.+?)__', r'\1', answer)               # __bold__
            answer = re.sub(r'\*(.+?)\*', r'\1', answer)               # *italic*
            answer = re.sub(r'_(.+?)_', r'\1', answer)                 # _italic_
            answer = re.sub(r'^#{1,6}\s+', '', answer, flags=re.MULTILINE)  # # headers
            answer = re.sub(r'^[-•]\s+', '• ', answer, flags=re.MULTILINE)  # bullets
            answer = re.sub(r'\n{3,}', '\n\n', answer)                 # excess blank lines
            answer = answer.strip()
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
    build_index(title, article["paragraphs"])
    return jsonify({
        "title": title,
        "url": article["url"],
        "table_count": article["table_count"],
        "chunk_count": len(article_store.get(title, {}).get("chunks", [])),
        "cached": False,
        "summary": article["paragraphs"][0] if article["paragraphs"] else "",
        "infobox": dict(list(article.get("infobox", {}).items())[:8]),
    })

@app.route("/api/ask", methods=["POST"])
def chat():
    data = request.json or {}
    query = (data.get("query") or data.get("title") or "").strip()
    question = (data.get("question") or "").strip()
    if not query or not question:
        return jsonify({"error": "query and question are required"}), 400

    logger.info(f"Q: {question} | Topic: {query}")
    title = resolve_title(query)
    if not title:
        return jsonify({"error": f"No Wikipedia article found for '{query}'"}), 404

    article = scrape_wikipedia(title)
    if "error" in article:
        return jsonify({"error": article["error"]}), 500

    build_index(title, article["paragraphs"])

    all_chunks = article_store.get(title, {}).get("chunks", [])
    if not all_chunks:
        return jsonify({"error": "No content found."}), 404

    top_chunks = retrieve_chunks_bm25(all_chunks, question, top_k=12)
    answer = answer_with_llm(question, top_chunks, title, article.get("infobox", {}))

    return jsonify({
        "query": query,
        "resolved_title": title,
        "answer": answer,
        "table_count": article["table_count"],
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