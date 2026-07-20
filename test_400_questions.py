"""
400-Question evaluation script for DeepQuery chatbot.
Tests all questions from the Excel sheet against the deployed API.

Usage:
    python test_400_questions.py

Requirements: pandas, openpyxl, requests
"""

import sys
import time
import json
import re

import pandas as pd
import requests

API_URL = "https://keshavgupta1511-deepqueryy-backend.hf.space/api/ask"
EXCEL = "Mount_Everest_RAG_Evaluation_400Q_Final 1.xlsx"
OUTPUT = "test_400_results.csv"
TOPIC = "Mount Everest"

REQUEST_DELAY = 2.0     # seconds between requests to avoid rate limits
RETRY_DELAY = 10         # seconds to wait before retrying on rate limit
MAX_RETRIES = 3

def ask(question: str) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(API_URL, json={"query": TOPIC, "question": question}, timeout=60)
            if r.status_code == 429:
                print(f"       [rate-limited, retrying in {RETRY_DELAY}s...]")
                time.sleep(RETRY_DELAY)
                continue
            if r.status_code != 200:
                return f"HTTP_ERROR: {r.status_code}"
            answer = r.json().get("answer", "NO_ANSWER_KEY")
            if "all free ai models are currently rate-limited" in answer.lower():
                if attempt < MAX_RETRIES:
                    print(f"       [model rate-limited, retrying in {RETRY_DELAY}s...]")
                    time.sleep(RETRY_DELAY)
                    continue
            return answer
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                print(f"       [timeout, retrying in {RETRY_DELAY}s...]")
                time.sleep(RETRY_DELAY)
                continue
            return "TIMEOUT"
        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"       [error: {e}, retrying in {RETRY_DELAY}s...]")
                time.sleep(RETRY_DELAY)
                continue
            return f"REQUEST_ERROR: {e}"
    return "MAX_RETRIES_EXCEEDED"

def is_valid(answer: str) -> bool:
    a = answer.strip().lower()
    if not a or a.startswith("http_error") or a == "timeout" or a.startswith("request_error"):
        return False
    if "all free ai models are currently rate-limited" in a:
        return False
    if "this question cannot be answered from the wikipedia article" in a:
        return False
    if "error" in a[:20] and len(a) < 100:
        return False
    return True

def main():
    print(f"Reading {EXCEL}...")
    df = pd.read_excel(EXCEL)
    required = ["S.No", "Question"]
    for col in required:
        if col not in df.columns:
            print(f"ERROR: Column '{col}' not found in Excel. Columns: {list(df.columns)}")
            sys.exit(1)

    total = len(df)
    results = []
    valid_count = 0
    start = time.time()

    print(f"Testing {total} questions against {API_URL}\n")

    has_category = "Category" in df.columns

    for idx, row in df.iterrows():
        sno = row["S.No"]
        question = str(row["Question"]).strip()
        if not question:
            continue

        answer = ask(question)
        valid = is_valid(answer)
        if valid:
            valid_count += 1

        time.sleep(REQUEST_DELAY)

        pct = valid_count / (idx + 1) * 100
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
        mark = "✓" if valid else "✗"
        preview = answer[:100].replace("\n", " ")
        print(f"[{sno}/{total}] {mark} {pct:5.1f}% | {elapsed} | {question[:50]}")
        print(f"       → {preview}")

        r = {"S.No": sno, "Question": question, "Answer": answer, "Valid": valid}
        if has_category:
            r["Category"] = row.get("Category", "")
        results.append(r)

        if (idx + 1) % 50 == 0:
            pd.DataFrame(results).to_csv(OUTPUT, index=False)
            print(f"       [saved intermediate: {idx+1}/{total}]\n")

    pd.DataFrame(results).to_csv(OUTPUT, index=False)
    elapsed_total = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))

    print("\n" + "=" * 60)
    print(f"OVERALL: {valid_count}/{total} = {valid_count/total*100:.1f}%")
    print(f"Time: {elapsed_total}")

    if has_category:
        rdf = pd.DataFrame(results)
        print("\n--- By Category ---")
        for cat, grp in rdf.groupby("Category"):
            cat_valid = grp["Valid"].sum()
            cat_total = len(grp)
            print(f"  {cat:35s} {cat_valid:3d}/{cat_total:3d} = {cat_valid/cat_total*100:5.1f}%")

    print(f"\nSaved to {OUTPUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
