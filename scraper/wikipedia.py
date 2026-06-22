import requests
from bs4 import BeautifulSoup

def scrape_wikipedia(url: str) -> str:
    """Scrapes raw text from a given Wikipedia URL, discarding styles and navigation text."""
    headers = {"User-Agent": "WikiRAGEngine/1.0 (contact: email@example.com)"}
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch Wikipedia page. Status code: {response.status_code}")
        
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Target only the main article text area
    content_div = soup.find(id="mw-content-text")
    if not content_div:
        raise Exception("Could not find main content text in the Wikipedia page.")
        
    # Extract paragraphs
    paragraphs = content_div.find_all("p")
    text_content = " ".join([p.get_text() for p in paragraphs if p.get_text().strip()])
    
    return text_content