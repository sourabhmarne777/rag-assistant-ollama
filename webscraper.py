import requests
from bs4 import BeautifulSoup

def scrape_urls(urls):
    texts = []
    for url in urls:
        try:
            res = requests.get(url, timeout=10)
            soup = BeautifulSoup(res.text, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            texts.append(text[:10000])  # limit to avoid overload
        except Exception as e:
            print(f"Error scraping {url}: {e}")
    return texts
