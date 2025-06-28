import requests
from bs4 import BeautifulSoup
from settings import MAX_TEXT_LENGTH
import logging

logger = logging.getLogger(__name__)

class WebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

    def scrape_url(self, url):
        """Scrape content from a URL with better content extraction"""
        try:
            logger.info(f"Scraping: {url}")
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
                element.decompose()
            
            # Get title
            title = soup.title.string if soup.title else url
            title = title.strip() if title else url
            
            # Try to find main content areas first
            content_selectors = [
                'main', 'article', '.content', '#content', '.post', '.entry',
                '.article-body', '.post-content', '.entry-content', 'section'
            ]
            
            main_content = None
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    main_content = elements[0]
                    break
            
            # If no main content found, use body
            if not main_content:
                main_content = soup.find('body') or soup
            
            # Extract text from main content
            text = main_content.get_text(separator=" ", strip=True)
            
            # Clean text
            lines = text.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line and len(line) > 10:  # Skip very short lines
                    cleaned_lines.append(line)
            
            text = " ".join(cleaned_lines)
            text = " ".join(text.split())  # Remove extra whitespace
            
            # Limit text length
            if len(text) > MAX_TEXT_LENGTH:
                text = text[:MAX_TEXT_LENGTH]
            
            # Check if we have enough meaningful content
            if len(text) < 200:  # Increased minimum content length
                logger.warning(f"Insufficient content from {url} (only {len(text)} chars)")
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
        """Scrape multiple URLs"""
        results = []
        for url in urls:
            if url.strip():
                result = self.scrape_url(url.strip())
                if result:
                    results.append(result)
        
        logger.info(f"Successfully scraped {len(results)} out of {len(urls)} URLs")
        return results

web_scraper = WebScraper()