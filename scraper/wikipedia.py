import requests
from bs4 import BeautifulSoup
from rapidfuzz import process

from scraper.cache import (
    get_cache,
    save_cache
)

HEADERS = {
    "User-Agent": "WikiRAGBot/1.0"
}


def resolve_title(query):

    url = (
        "https://en.wikipedia.org/w/api.php"
        "?action=opensearch"
        f"&search={query}"
        "&limit=10"
        "&format=json"
    )

    response = requests.get(
        url,
        headers=HEADERS
    )

    candidates = response.json()[1]

    if not candidates:
        return None

    match = process.extractOne(
        query,
        candidates
    )

    return match[0]


def scrape_article(title):

    cached = get_cache(title)

    if cached:
        print("Cache Hit:", title)
        return cached

    print("Scraping:", title)

    url = (
        "https://en.wikipedia.org/wiki/"
        + title.replace(" ", "_")
    )

    html = requests.get(
        url,
        headers=HEADERS
    ).text

    soup = BeautifulSoup(
        html,
        "lxml"
    )

    tables = soup.find_all("table")

    paragraphs = soup.select(
        "div.mw-parser-output > p"
    )

    text = []

    for p in paragraphs:

        t = p.get_text(
            " ",
            strip=True
        )

        if len(t) > 50:
            text.append(t)

    data = {
        "text": "\n".join(text),
        "tables": len(tables)
    }

    save_cache(
        title,
        data
    )

    return data