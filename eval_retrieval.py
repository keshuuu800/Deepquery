"""
Retrieval Accuracy Evaluation Script

Measures how well the RAG system retrieves relevant passages for a set of
test queries. For each query, we check if the correct answer can be found
within the top-k retrieved chunks.

Usage:
    python eval_retrieval.py              # Run evaluation
    python eval_retrieval.py --verbose    # Show per-query details
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from app import (
    resolve_title,
    scrape_wikipedia,
    build_index,
    expand_and_retrieve,
    rerank_chunks,
    retrieve_chunks_bm25,
    article_store,
)

TEST_CASES = [
    # (topic, question, expected_keywords)
    ("Mount Everest", "How tall is Mount Everest?", ["8848", "8,848", "29,029", "meters", "feet"]),
    ("Mount Everest", "Who first climbed Everest?", ["Hillary", "Norgay", "Tenzing"]),
    ("Albert Einstein", "What is Einstein's theory of relativity?", ["relativity", "E=mc", "general", "special"]),
    ("Albert Einstein", "When was Einstein born?", ["1879"]),
    ("Python (programming language)", "Who created Python?", ["Guido", "van Rossum"]),
    ("Python (programming language)", "What year was Python released?", ["1991"]),
    ("Mars", "What is the temperature on Mars?", ["temperature", "cold", "degrees", "Celsius"]),
    ("Mars", "How many moons does Mars have?", ["2", "two", "Phobos", "Deimos"]),
    ("Great Wall of China", "How long is the Great Wall of China?", ["21,000", "13,000", "miles", "kilometers", "km"]),
    ("Great Wall of China", "When was the Great Wall built?", ["century", "BC", "Ming", "dynasty"]),
    ("Quantum mechanics", "What is Schrödinger's cat?", ["Schrödinger", "cat", "quantum", "superposition"]),
    ("World War II", "When did World War II end?", ["1945"]),
    ("World War II", "Who was the US president during WWII?", ["Roosevelt", "Truman"]),
    ("Amazon rainforest", "Where is the Amazon rainforest located?", ["South America", "Brazil", "Amazon basin"]),
    ("Amazon rainforest", "What percentage of Earth's oxygen does the Amazon produce?", ["20", "6", "percent"]),
    ("Oxygen", "What is the chemical symbol for oxygen?", ["O"]),
    ("Oxygen", "Who discovered oxygen?", ["Priestley", "Scheele", "Lavoisier"]),
]


def evaluate(top_k: int = 10):
    results = []
    total, hits_at_1, hits_at_5, hits_at_10 = 0, 0, 0, 0

    for topic, question, keywords in TEST_CASES:
        total += 1
        title = resolve_title(topic)
        if not title:
            print(f"  ✗ FAIL: Could not resolve title for '{topic}'")
            results.append((topic, question, "TITLE_FAIL", False, False, False))
            continue

        article = scrape_wikipedia(title)
        if "error" in article:
            print(f"  ✗ FAIL: Scrape error for '{title}': {article['error']}")
            results.append((topic, question, "SCRAPE_FAIL", False, False, False))
            continue

        build_index(title, article["paragraphs"], article.get("table_chunks", []))
        entry = article_store.get(title, {})
        chunks = entry.get("chunks", [])
        embeddings = entry.get("embeddings")

        if embeddings is not None:
            retrieved = expand_and_retrieve(chunks, embeddings, question, top_k=16)
            retrieved = rerank_chunks(question, retrieved, top_k=10)
        else:
            retrieved = retrieve_chunks_bm25(chunks, question, top_k=10)

        found_any = False
        found_in_top1 = False
        found_in_top5 = False
        matched_keywords = []

        for kw in keywords:
            kw_lower = kw.lower()
            for i, chunk in enumerate(retrieved):
                if kw_lower in chunk.lower():
                    matched_keywords.append(kw)
                    if i == 0:
                        found_in_top1 = True
                    if i < 5:
                        found_in_top5 = True
                    found_any = True
                    break

        if found_any:
            hits_at_10 += 1
        if found_in_top5:
            hits_at_5 += 1
        if found_in_top1:
            hits_at_1 += 1

        status = "✓" if found_any else "✗"
        top1 = retrieved[0][:80].replace("\n", " ") if retrieved else "(no chunks)"
        print(
            f"  {status} Q: {question[:60]}"
            f"\n    Top-1: {top1}..."
            f"\n    Keywords matched: {matched_keywords}"
            f"\n    Hit@1={found_in_top1} Hit@5={found_in_top5} Hit@10={found_any}"
        )

        results.append((topic, question, retrieved, found_in_top1, found_in_top5, found_any))

    print("\n" + "=" * 60)
    print(f"RESULTS over {total} queries (top_k={top_k}):")
    print(f"  Hit@1  = {hits_at_1}/{total} = {hits_at_1/total*100:.1f}%")
    print(f"  Hit@5  = {hits_at_5}/{total} = {hits_at_5/total*100:.1f}%")
    print(f"  Hit@10 = {hits_at_10}/{total} = {hits_at_10/total*100:.1f}%")
    print("=" * 60)

    return {
        "total": total,
        "hit@1": hits_at_1,
        "hit@5": hits_at_5,
        "hit@10": hits_at_10,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval accuracy")
    parser.add_argument("--top-k", type=int, default=10, help="Number of chunks to retrieve")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-query details")
    args = parser.parse_args()
    evaluate(top_k=args.top_k)
