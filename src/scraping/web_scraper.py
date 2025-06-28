import requests
from bs4 import BeautifulSoup
from config.settings import MAX_TEXT_LENGTH
import logging

logger = logging.getLogger(__name__)

class WebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def scrape_url(self, url):
        """Scrape a single URL and return cleaned text"""
        try:
            logger.info(f"Scraping URL: {url}")
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
                element.decompose()
            
            # Extract title
            title = soup.title.string if soup.title else url
            title = title.strip() if title else url
            
            # Extract main content
            text = soup.get_text(separator=" ", strip=True)
            
            # Clean and limit text
            text = " ".join(text.split())  # Remove extra whitespace
            text = text[:MAX_TEXT_LENGTH]
            
            if len(text) < 50:  # Skip if too little content
                logger.warning(f"Insufficient content from {url}")
                return None
            
            logger.info(f"Successfully scraped {len(text)} characters from {url}")
            return {
                'url': url,
                'content': text,
                'title': title
            }
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None

    def scrape_urls(self, urls):
        """Scrape multiple URLs and return results"""
        results = []
        for url in urls:
            if url.strip():
                result = self.scrape_url(url.strip())
                if result:  # Only add if content was scraped successfully
                    results.append(result)
        
        logger.info(f"Successfully scraped {len(results)} out of {len(urls)} URLs")
        return results

# Global instance
web_scraper = WebScraper()
