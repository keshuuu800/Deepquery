from scraper.wikipedia import (
    resolve_title,
    scrape_article
)

title = resolve_title(
    "Akshay Kumar"
)

data = scrape_article(title)

print(data["tables"])