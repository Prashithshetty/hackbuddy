import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict
import re
import feedparser
import html2text
import json

"""
Problem scraper module for collecting hackathon problem statements.
Fetches problems from multiple sources including:
- arXiv research papers
- GitHub repositories
- Tech blogs via RSS feeds

Handles different API formats and response processing with error handling.
"""

class ProblemScraper:
    """Multi-source problem scraper for hackathon challenges.
    
    Fetches potential hackathon problems from various online sources,
    processes them into a consistent format, and estimates difficulty.
    
    Attributes:
        sources (dict): Configuration for different data sources including:
            - URL templates
            - Data extraction methods
        html_converter (HTML2Text): HTML to plain text converter
    """
    def __init__(self):
        self.sources = {
            "arxiv": {
                "url": f"http://export.arxiv.org/api/query?search_query=cat:{{}}&sortBy=lastUpdatedDate&max_results=5",
                "extractor": self._extract_arxiv_problems
            },
            "github": {
                "url": "https://api.github.com/search/repositories?q=topic:{}+is:public",
                "extractor": self._extract_github_api_problems
            },
            "tech_blogs": {
                "url": "https://hnrss.org/newest?q={}",
                "extractor": self._extract_rss_problems
            }
        }
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = True
        
    def scrape_problems(self, theme: str) -> List[Dict]:
        """Scrape problems matching theme from all configured sources.
        
        Concurrently fetches problems from multiple sources, processes them,
        and returns unified problem descriptions.
        
        Args:
            theme (str): Theme to search for (e.g., "AI", "Healthcare")
            
        Returns:
            List[Dict]: List of problems with standardized structure:
                - title: Problem title
                - description: Detailed description
                - theme: Original search theme
                - source_url: Source location
                - difficulty: Estimated difficulty level
        """
        problems = []
        theme_query = theme.lower().replace(' ', '+')
        
        for source_name, source_info in self.sources.items():
            try:
                url = source_info["url"].format(theme_query)
                response = requests.get(url, headers={
                    'User-Agent': 'Mozilla/5.0',
                    'Accept': 'application/json'
                }, timeout=10)
                
                if response.status_code == 200:
                    source_problems = source_info["extractor"](response.text, theme)
                    if source_problems:
                        problems.extend(source_problems[:2])
                    logging.info(f"Scraped {len(source_problems)} problems from {source_name}")
            except Exception as e:
                logging.error(f"Error scraping {source_name}: {str(e)}")
        
        return problems[:5]

    def _extract_arxiv_problems(self, content: str, theme: str) -> List[Dict]:
        """Extract problems from arXiv API response.
        
        Processes arXiv paper metadata into problem statements,
        estimates difficulty based on content complexity.
        
        Args:
            content (str): Raw arXiv API response
            theme (str): Current theme for categorization
            
        Returns:
            List[Dict]: Extracted problems in standard format
        """
        problems = []
        try:
            feed = feedparser.parse(content)
            for entry in feed.entries:
                problems.append({
                    'title': entry.title,
                    'description': self.html_converter.handle(entry.summary),
                    'theme': theme,
                    'source_url': entry.link,
                    'difficulty': self._estimate_difficulty(entry.summary)
                })
        except Exception as e:
            logging.error(f"Error parsing arXiv content: {str(e)}")
        return problems

    def _extract_github_api_problems(self, content: str, theme: str) -> List[Dict]:
        """Extract problems from GitHub repository search results.
        
        Converts GitHub repositories into potential hackathon problems
        based on repository descriptions and metadata.
        
        Args:
            content (str): GitHub API response JSON
            theme (str): Current theme for categorization
            
        Returns:
            List[Dict]: Extracted problems in standard format
        """
        problems = []
        try:
            data = json.loads(content)
            for repo in data.get('items', []):
                problems.append({
                    'title': repo['name'],
                    'description': repo['description'] or "No description available",
                    'theme': theme,
                    'source_url': repo['html_url'],
                    'difficulty': 'Medium'
                })
        except Exception as e:
            logging.error(f"Error parsing GitHub API content: {str(e)}")
        return problems

    def _extract_rss_problems(self, content: str, theme: str) -> List[Dict]:
        """Extract problems from RSS feed entries.
        
        Processes blog posts and articles into problem statements,
        focusing on technical challenges and solutions.
        
        Args:
            content (str): Raw RSS feed content
            theme (str): Current theme for categorization
            
        Returns:
            List[Dict]: Extracted problems in standard format
        """
        problems = []
        try:
            feed = feedparser.parse(content)
            for entry in feed.entries:
                problems.append({
                    'title': entry.title,
                    'description': self.html_converter.handle(entry.summary),
                    'theme': theme,
                    'source_url': entry.link,
                    'difficulty': 'Medium'
                })
        except Exception as e:
            logging.error(f"Error parsing RSS content: {str(e)}")
        return problems

    def _estimate_difficulty(self, description: str) -> str:
        """Estimate problem difficulty based on description complexity.
        
        Uses text length and complexity metrics to categorize difficulty.
        
        Args:
            description (str): Problem description text
            
        Returns:
            str: Difficulty level ('Easy', 'Medium', or 'Hard')
        """
        word_count = len(description.split())
        if word_count > 300:
            return 'Hard'
        elif word_count > 150:
            return 'Medium'
        return 'Easy'
