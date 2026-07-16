from app import resolve_title, scrape_wikipedia

title = resolve_title("Akshay Kumar")
if title:
    data = scrape_wikipedia(title)
    print(f"Title: {title}")
    print(f"Paragraphs: {len(data.get('paragraphs', []))}")
    print(f"Table chunks: {len(data.get('table_chunks', []))}")
    if data.get("table_chunks"):
        print(f"First table chunk preview: {data['table_chunks'][0][:200]}")
else:
    print("Could not resolve title")
