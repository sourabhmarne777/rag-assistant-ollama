import requests
from bs4 import BeautifulSoup
from typing import Optional
import re

class WebScraper:
    """Web scraper for extracting content from URLs"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.timeout = 10
    
    def scrape_url(self, url: str) -> Optional[str]:
        """Scrape content from a given URL"""
        try:
            # Validate URL
            if not self._is_valid_url(url):
                raise ValueError("Invalid URL format")
            
            # Make request
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text content
            content = self._extract_content(soup, url)
            
            return content
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL {url}: {e}")
            return None
        except Exception as e:
            print(f"Error processing URL {url}: {e}")
            return None
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return url_pattern.match(url) is not None
    
    def _extract_content(self, soup: BeautifulSoup, url: str) -> str:
        """Extract meaningful content from HTML"""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
            element.decompose()
        
        # Try to find main content areas
        content_selectors = [
            'article',
            'main',
            '[role="main"]',
            '.content',
            '.post-content',
            '.entry-content',
            '.article-content',
            '#content',
            '#main-content'
        ]
        
        main_content = None
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                main_content = elements[0]
                break
        
        # If no main content found, use body
        if main_content is None:
            main_content = soup.find('body') or soup
        
        # Extract text
        text_content = main_content.get_text(separator='\n', strip=True)
        
        # Clean up the text
        text_content = self._clean_text(text_content)
        
        # Add metadata
        title = soup.find('title')
        title_text = title.get_text().strip() if title else "No title"
        
        # Combine title and content
        full_content = f"Title: {title_text}\nURL: {url}\n\nContent:\n{text_content}"
        
        return full_content
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove very short lines (likely navigation or ads)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) > 10:  # Keep lines with substantial content
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def get_page_metadata(self, url: str) -> dict:
        """Extract metadata from a webpage"""
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            metadata = {
                'url': url,
                'title': '',
                'description': '',
                'keywords': '',
                'author': '',
                'published_date': ''
            }
            
            # Extract title
            title = soup.find('title')
            if title:
                metadata['title'] = title.get_text().strip()
            
            # Extract meta tags
            meta_tags = soup.find_all('meta')
            for tag in meta_tags:
                name = tag.get('name', '').lower()
                property_attr = tag.get('property', '').lower()
                content = tag.get('content', '')
                
                if name == 'description' or property_attr == 'og:description':
                    metadata['description'] = content
                elif name == 'keywords':
                    metadata['keywords'] = content
                elif name == 'author':
                    metadata['author'] = content
                elif property_attr == 'article:published_time':
                    metadata['published_date'] = content
            
            return metadata
            
        except Exception as e:
            print(f"Error extracting metadata from {url}: {e}")
            return {'url': url, 'error': str(e)}