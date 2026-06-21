"""
Wikipedia RAG Chatbot Backend
Scrapes Wikipedia with BeautifulSoup, builds embeddings, answers via LLM API.
FIX: Smart model rotation + exponential backoff for OpenRouter 429 errors.
"""

import os, json, time, hashlib, re, logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer
import numpy as np
from openai import OpenAI, RateLimitError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

HEADERS = {"User-Agent": "WikiRAGBot/1.0 (Educational project) requests/2.x"}
RATE_LIMIT_DELAY = 0.8

logger.info("Loading embedding model…")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
logger.info("Embedding model ready.")

vector_store: dict = {}

# ─────────────────────────────────────────────────────────────
#  FREE MODEL ROTATION LIST
#  All free on OpenRouter — rotated in order when 429 hits
# ─────────────────────────────────────────────────────────────
FREE_MODELS = [
    "openai/gpt-oss-20b:free",
    "meta-llama/llama-3.3-70b-instruct:free"
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

# ── Cache helpers ──────────────────────────────────────────────

def _ck(text: str) -> str:
    return hashlib.md5(text.lower().strip().encode()).hexdigest()

def cache_get(key: str) -> Optional[dict]:
    p = CACHE_DIR / f"{key}.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None

def cache_set(key: str, data: dict):
    p = CACHE_DIR / f"{key}.json"
    with open(p, "w") as f:
        json.dump(data, f, ensure_ascii=False)

# ── Wikipedia helpers ──────────────────────────────────────────

def wikipedia_search(query: str) -> list:
    url = "https://en.wikipedia.org/w/api.php"
    params = {"action": "opensearch", "search": query, "limit": 10,
               "namespace": 0, "format": "json"}
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
    title = best[0] if (best and best[1] >= 70) else candidates[0]
    cache_set(ck, {"title": title})
    time.sleep(RATE_LIMIT_DELAY)
    return title

def scrape_wikipedia(title: str) -> dict:
    ck = _ck(f"scrape_{title}")
    cached = cache_get(ck)
    if cached:
        logger.info(f"Cache hit: {title}")
        return cached

    url_title = title.replace(" ", "_")
    url = f"https://en.wikipedia.org/wiki/{url_title}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        time.sleep(RATE_LIMIT_DELAY)
    except Exception as e:
        logger.error(f"Scrape error {title}: {e}")
        return {"error": str(e)}

    soup = BeautifulSoup(resp.text, "html.parser")
    table_count = len(soup.find_all("table"))

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
        "full_text": "\n\n".join(paragraphs),
        "paragraphs": paragraphs,
        "table_count": table_count,
        "infobox": infobox_data,
    }
    cache_set(ck, result)
    return result

# ── RAG pipeline ───────────────────────────────────────────────

def chunk_text(paragraphs: list, max_words: int = 350) -> list:
    chunks, cur, cur_len = [], [], 0
    for para in paragraphs:
        words = para.split()
        if cur_len + len(words) > max_words and cur:
            chunks.append(" ".join(cur))
            cur = cur[-40:]
            cur_len = len(cur)
        cur.extend(words)
        cur_len += len(words)
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def build_index(title: str, paragraphs: list):
    if title in vector_store:
        return
    ck = _ck(f"emb_{title}")
    cached = cache_get(ck)
    if cached:
        vector_store[title] = {
            "chunks": cached["chunks"],
            "embeddings": np.array(cached["embeddings"]),
        }
        logger.info(f"Loaded embeddings from cache: {title}")
        return
    chunks = chunk_text(paragraphs)
    if not chunks:
        return
    logger.info(f"Encoding {len(chunks)} chunks for {title}…")
    embs = embedder.encode(chunks, show_progress_bar=False)
    vector_store[title] = {"chunks": chunks, "embeddings": embs}
    cache_set(ck, {"chunks": chunks, "embeddings": embs.tolist()})

def retrieve_chunks(title: str, question: str, top_k: int = 5) -> list:
    if title not in vector_store:
        return []
    q_emb = embedder.encode([question])
    store = vector_store[title]
    embs = store["embeddings"]
    q_n = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)
    d_n = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)
    scores = (q_n @ d_n.T)[0]
    idxs = np.argsort(scores)[::-1][:top_k]
    return [store["chunks"][i] for i in idxs]

# ── LLM with model rotation + retry ────────────────────────────

def answer_with_llm(question: str, chunks: list, title: str, infobox: dict) -> str:
    # Check cache first — avoid unnecessary API calls
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

    prompt = f"""You are a helpful assistant.
Use ONLY the Wikipedia context below about "{title}".
If the answer is not present in the context, reply: "Information not found in Wikipedia article."

{ib}CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    last_error = None

    for model_idx, model in enumerate(FREE_MODELS):
        # Exponential backoff: wait longer for later models (they're fallbacks)
        if model_idx > 0:
            wait = min(2 ** model_idx, 16)  # 2s, 4s, 8s, 16s max
            logger.info(f"Waiting {wait}s before trying {model}…")
            time.sleep(wait)

        try:
            logger.info(f"Trying model: {model}")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=600,
                timeout=45,
            )
            answer = response.choices[0].message.content.strip()
            logger.info(f"✓ Got answer from {model}")

            # Cache the successful answer
            cache_set(qa_key, {"answer": answer, "model": model})
            return answer

        except RateLimitError as e:
            last_error = e
            logger.warning(f"429 on {model} — rotating to next model")
            continue  # try next model immediately

        except Exception as e:
            last_error = e
            logger.error(f"Error on {model}: {e}")
            continue  # try next model

    # All models exhausted
    logger.error(f"All {len(FREE_MODELS)} models failed. Last error: {last_error}")
    return (
        "⚠ All free AI models are currently rate-limited. "
        "Please wait 30–60 seconds and try again, or add your own OpenRouter API key "
        "to get higher rate limits at openrouter.ai/settings/integrations."
    )

# ── Routes ─────────────────────────────────────────────────────

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
        "chunk_count": len(vector_store.get(title, {}).get("chunks", [])),
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

    logger.info(f"QUERY: {query} | QUESTION: {question}")

    title = resolve_title(query)
    if not title:
        return jsonify({"error": f"No Wikipedia article found for '{query}'"}), 404

    article = scrape_wikipedia(title)
    if "error" in article:
        return jsonify({"error": article["error"]}), 500

    build_index(title, article["paragraphs"])

    chunks = retrieve_chunks(title, question, top_k=10)
    if not chunks:
        return jsonify({"error": "No relevant content found."}), 404

    answer = answer_with_llm(question, chunks, title, article.get("infobox", {}))

    return jsonify({
        "query": query,
        "resolved_title": title,
        "answer": answer,
        "table_count": article["table_count"],
        "chunks_used": len(chunks),
        "article_url": article["url"],
        "cached": False,
    })

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "models_available": len(FREE_MODELS)})

if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=8000
    )