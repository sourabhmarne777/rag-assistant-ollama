import requests
from bs4 import BeautifulSoup
from typing import Optional, Dict, Any
import logging
from urllib.parse import urljoin, urlparse
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebScraper:
    """Web scraper for extracting content from URLs"""
    
    def __init__(self, timeout: int = 30, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Headers to mimic a real browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def scrape_url(self, url: str) -> Optional[str]:
        """Scrape content from a URL"""
        try:
            # Validate URL
            if not self._is_valid_url(url):
                raise ValueError(f"Invalid URL: {url}")
            
            # Make request with retries
            response = self._make_request_with_retries(url)
            
            if response is None:
                return None
            
            # Parse content
            content = self._extract_content(response.text, url)
            
            if content:
                logger.info(f"Successfully scraped content from {url}")
                return content
            else:
                logger.warning(f"No content extracted from {url}")
                return None
                
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {str(e)}")
            return None
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def _make_request_with_retries(self, url: str) -> Optional[requests.Response]:
        """Make HTTP request with retries"""
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    url,
                    headers=self.headers,
                    timeout=self.timeout,
                    allow_redirects=True
                )
                
                # Check if request was successful
                response.raise_for_status()
                
                return response
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    # Wait before retrying
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"All {self.max_retries} attempts failed for {url}")
                    return None
    
    def _extract_content(self, html: str, url: str) -> Optional[str]:
        """Extract meaningful content from HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove unwanted elements
            self._remove_unwanted_elements(soup)
            
            # Try different strategies to extract main content
            content = self._extract_main_content(soup)
            
            if not content:
                # Fallback: extract all text
                content = soup.get_text(separator='\n', strip=True)
            
            # Clean and validate content
            content = self._clean_content(content)
            
            if len(content.strip()) < 100:  # Too short to be meaningful
                logger.warning(f"Extracted content too short from {url}")
                return None
            
            return content
            
        except Exception as e:
            logger.error(f"Error extracting content from HTML: {str(e)}")
            return None
    
    def _remove_unwanted_elements(self, soup: BeautifulSoup):
        """Remove unwanted HTML elements"""
        # Elements to remove
        unwanted_tags = [
            'script', 'style', 'nav', 'header', 'footer', 
            'aside', 'advertisement', 'ads', 'sidebar',
            'menu', 'breadcrumb', 'social', 'share'
        ]
        
        for tag in unwanted_tags:
            for element in soup.find_all(tag):
                element.decompose()
        
        # Remove elements with unwanted classes/ids
        unwanted_patterns = [
            'nav', 'menu', 'sidebar', 'footer', 'header',
            'ad', 'advertisement', 'social', 'share',
            'comment', 'related', 'popup', 'modal'
        ]
        
        for pattern in unwanted_patterns:
            # Remove by class
            for element in soup.find_all(class_=lambda x: x and pattern in str(x).lower()):
                element.decompose()
            
            # Remove by id
            for element in soup.find_all(id=lambda x: x and pattern in str(x).lower()):
                element.decompose()
    
    def _extract_main_content(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract main content using various strategies"""
        content_selectors = [
            # Common article/content selectors
            'article',
            '[role="main"]',
            'main',
            '.content',
            '.post-content',
            '.entry-content',
            '.article-content',
            '.story-body',
            '.post-body',
            '#content',
            '#main-content',
            '.main-content'
        ]
        
        for selector in content_selectors:
            try:
                elements = soup.select(selector)
                if elements:
                    # Get the largest element (most likely to be main content)
                    main_element = max(elements, key=lambda x: len(x.get_text()))
                    content = main_element.get_text(separator='\n', strip=True)
                    
                    if len(content.strip()) > 200:  # Reasonable content length
                        return content
            except Exception:
                continue
        
        # Fallback: look for the largest text block
        try:
            all_elements = soup.find_all(['p', 'div', 'section', 'article'])
            if all_elements:
                largest_element = max(all_elements, key=lambda x: len(x.get_text()))
                content = largest_element.get_text(separator='\n', strip=True)
                
                if len(content.strip()) > 200:
                    return content
        except Exception:
            pass
        
        return None
    
    def _clean_content(self, content: str) -> str:
        """Clean extracted content"""
        try:
            # Split into lines and clean each line
            lines = content.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line = line.strip()
                
                # Skip empty lines and very short lines
                if len(line) < 3:
                    continue
                
                # Skip lines that look like navigation or metadata
                if self._is_navigation_line(line):
                    continue
                
                cleaned_lines.append(line)
            
            # Join lines back together
            cleaned_content = '\n'.join(cleaned_lines)
            
            # Remove excessive whitespace
            cleaned_content = ' '.join(cleaned_content.split())
            
            return cleaned_content
            
        except Exception as e:
            logger.error(f"Error cleaning content: {str(e)}")
            return content
    
    def _is_navigation_line(self, line: str) -> bool:
        """Check if a line looks like navigation or metadata"""
        line_lower = line.lower()
        
        # Common navigation/metadata patterns
        nav_patterns = [
            'home', 'about', 'contact', 'privacy', 'terms',
            'login', 'register', 'sign up', 'sign in',
            'menu', 'navigation', 'breadcrumb',
            'share', 'tweet', 'facebook', 'linkedin',
            'subscribe', 'newsletter', 'follow us',
            'copyright', '©', 'all rights reserved',
            'cookies', 'gdpr'
        ]
        
        # Check if line contains navigation patterns
        for pattern in nav_patterns:
            if pattern in line_lower:
                return True
        
        # Check if line is mostly punctuation or numbers
        if len([c for c in line if c.isalnum()]) < len(line) * 0.5:
            return True
        
        return False
    
    def get_page_metadata(self, url: str) -> Dict[str, Any]:
        """Extract metadata from a webpage"""
        try:
            response = self._make_request_with_retries(url)
            if response is None:
                return {}
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            metadata = {
                'url': url,
                'title': '',
                'description': '',
                'author': '',
                'published_date': ''
            }
            
            # Extract title
            title_tag = soup.find('title')
            if title_tag:
                metadata['title'] = title_tag.get_text().strip()
            
            # Extract meta description
            desc_tag = soup.find('meta', attrs={'name': 'description'})
            if desc_tag:
                metadata['description'] = desc_tag.get('content', '').strip()
            
            # Extract author
            author_tag = soup.find('meta', attrs={'name': 'author'})
            if author_tag:
                metadata['author'] = author_tag.get('content', '').strip()
            
            # Extract published date
            date_selectors = [
                'meta[property="article:published_time"]',
                'meta[name="publish-date"]',
                'meta[name="date"]',
                'time[datetime]'
            ]
            
            for selector in date_selectors:
                date_element = soup.select_one(selector)
                if date_element:
                    date_value = date_element.get('content') or date_element.get('datetime')
                    if date_value:
                        metadata['published_date'] = date_value.strip()
                        break
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {url}: {str(e)}")
            return {'url': url}