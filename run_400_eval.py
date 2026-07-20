"""
Full 400-question evaluation against the chatbot.
Measures how many questions get a valid answer (not "not found" or rate-limited).
"""
import os, sys, json, time, re
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from app import (
    resolve_title, scrape_wikipedia, build_index, article_store,
    hybrid_retrieve, answer_with_llm, cache_get, cache_set, _ck
)

EXCEL = os.path.join(os.path.dirname(__file__),
                     "Mount_Everest_RAG_Evaluation_400Q_Final 1.xlsx")

def answer_question(topic: str, question: str) -> str:
    title = resolve_title(topic)
    if not title:
        return "ERROR: Could not resolve title"
    article = scrape_wikipedia(title)
    if "error" in article:
        return f"ERROR: {article['error']}"
    build_index(title, article["paragraphs"], article.get("table_chunks", []))
    entry = article_store.get(title, {})
    chunks = entry.get("chunks", [])
    embeddings = entry.get("embeddings")
    if not chunks:
        return "ERROR: No chunks"
    top = hybrid_retrieve(chunks, embeddings, question, title=title, top_k=15) if embeddings is not None \
          else retrieve_chunks_bm25(chunks, question, top_k=15)
    answer = answer_with_llm(question, top, title, article.get("infobox", {}))
    return answer

def is_valid(answer: str) -> bool:
    a = answer.strip().lower()
    if a.startswith("error"):
        return False
    if "information not found" in a:
        return False
    if "all free ai models are currently rate-limited" in a:
        return False
    return True

def main():
    df = pd.read_excel(EXCEL)
    results = []
    valid, total = 0, 0
    start = time.time()

    for _, row in df.iterrows():
        total += 1
        sno = row["S.No"]
        question = row["Question"]
        topic = "Mount Everest"

        # Cache check to save LLM calls
        qa_key = _ck(f"qa_{topic}_{question}")
        cached = cache_get(qa_key)
        if cached and cached.get("answer"):
            answer = cached["answer"]
            used_cache = True
        else:
            answer = answer_question(topic, question)
            used_cache = False

        valid_flag = is_valid(answer)
        if valid_flag:
            valid += 1

        results.append({
            "S.No": sno,
            "Question": question,
            "Answer": answer[:120],
            "Valid": valid_flag,
            "Cached": used_cache,
        })

        elapsed = time.time() - start
        rate = valid / total * 100
        print(f"[{sno}/400] {'✓' if valid_flag else '✗'} rate={rate:.0f}% | {time.strftime('%H:%M:%S', time.gmtime(elapsed))} | {question[:60]}")
        sys.stdout.flush()

        if total % 10 == 0:
            # Save intermediate results
            pd.DataFrame(results).to_csv("eval_intermediate.csv", index=False)

    print("\n" + "=" * 60)
    print(f"RESULTS: {valid}/{total} = {valid/total*100:.1f}% valid answers")
    print(f"Time: {time.strftime('%H:%M:%S', time.gmtime(time.time()-start))}")
    pd.DataFrame(results).to_csv("eval_final.csv", index=False)
    print("Saved to eval_final.csv")

if __name__ == "__main__":
    main()
