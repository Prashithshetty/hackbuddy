"""
Main application for generating AI-powered hackathon problems.
Integrates web crawling, content analysis, and AI-based problem generation
using the LLaMA model. Provides an interactive interface for generating
themed hackathon challenges with difficulty estimation and evaluation metrics.

Key components:
- WebCrawler: Asynchronous content fetching with caching
- ThematicAnalyzer: Theme-based content analysis and problem generation
- ProblemDiscoverer: Source management and category handling
"""

from llama_cpp import Llama
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser
from bs4 import BeautifulSoup
import concurrent.futures
import logging
import os
import json
from datetime import datetime
import re
import ssl
import time
from typing import List, Dict
import random
import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed
from cachetools import TTLCache, LRUCache
from functools import lru_cache
import aiohttp
import asyncio
import hashlib

# Add imports
from database import DatabaseHandler
from scraper import ProblemScraper

def setup_logging():
    """Configure application-wide logging.
    
    Sets up both file and console logging handlers with formatting.
    Log file: 'hackathon_generator.log'
    
    Returns:
        Logger: Configured logging instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('hackathon_generator.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class WebCrawler:
    """Asynchronous web crawler with caching and rate limiting.
    
    Handles content fetching with:
    - SSL context management
    - Rate limiting per domain
    - Content caching (TTL: 1 hour)
    - Robots.txt compliance
    - Main content extraction
    
    Attributes:
        visited (set): Track crawled URLs
        max_depth (int): Maximum crawl depth
        max_pages (int): Maximum pages to crawl
        content_cache (TTLCache): Page content cache
        session (aiohttp.ClientSession): Async HTTP session
    """
    
    def __init__(self, max_depth=2, max_pages=5):
        self.visited = set()
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.context = ssl._create_unverified_context()
        self.rate_limits = {}
        self.session = None
        self.content_cache = TTLCache(maxsize=100, ttl=3600)  # Cache for 1 hour
        
    async def init_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
            
    async def close_session(self):
        if self.session:
            await self.session.close()
            
    def cache_key(self, url):
        return hashlib.md5(url.encode()).hexdigest()

    async def get_page_content(self, url):
        """Enhanced async page content fetching with caching"""
        cache_key = self.cache_key(url)
        
        if cache_key in self.content_cache:
            return self.content_cache[cache_key]
            
        if not self.respect_robots(url):
            return ""
            
        domain = urlparse(url).netloc
        self.rate_limit(domain)
        
        try:
            await self.init_session()
            async with self.session.get(url, ssl=self.context) as response:
                if response.status != 200:
                    return ""
                
                content = await response.text()
                soup = BeautifulSoup(content, 'html.parser')
                
                # Optimize content extraction
                [tag.decompose() for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside'])]
                
                # Find main content more efficiently
                main_content = next((soup.select_one(selector) for selector in [
                    'article', 'main', '[role="main"]', '.post-content',
                    '.article-content', '#content-body'
                ] if soup.select_one(selector)), soup)
                
                # Efficient text extraction
                paragraphs = [p.get_text(strip=True) for p in main_content.find_all(['p', 'h1', 'h2', 'h3', 'li'])]
                text = ' '.join(p for p in paragraphs if len(p) > 30)
                
                # Efficient text cleaning
                text = re.sub(r'\s+', ' ', text)
                text = re.sub(r'[^\w\s.,!?-]', '', text)
                
                result = text[:8000]
                self.content_cache[cache_key] = result
                return result
                
        except Exception as e:
            logging.error(f"Error fetching {url}: {str(e)}")
            return ""

    async def crawl_theme_sources(self, theme: str) -> List[Dict]:
        """Concurrent crawling of theme sources"""
        theme_data = self.themes.get(theme, {})
        sources = theme_data.get('sources', [])
        
        async def crawl_source(source):
            content = await self.get_page_content(source)
            if content:
                return {
                    'url': source,
                    'content': content,
                    'timestamp': datetime.now().isoformat()
                }
            return None
            
        tasks = [crawl_source(source) for source in sources]
        results = []
        
        for task in asyncio.as_completed(tasks):
            result = await task
            if result:
                results.append(result)
                
        return results

class ProblemDiscoverer:
    """Problem domain and category manager.
    
    Maintains curated lists of:
    - Problem sources (URLs)
    - Problem categories
    - Category-specific prompts
    
    Used for organizing and categorizing generated problems.
    """
    
    def __init__(self):
        self.problem_sources = [
            "https://www.un.org/sustainabledevelopment/sustainable-development-goals/",
            "https://www.who.int/health-topics/",
            "https://challenges.org",
            "https://www.climate.gov/teaching/climate-challenges",
            "https://www.challenge.gov",
            # Add more reliable sources
        ]
        
        self.categories = [
            "Healthcare",
            "Education",
            "Climate Change",
            "Transportation",
            "Agriculture",
            "Business",
            "Logistics",
            "Social Impact",
            "Financial Inclusion",
            "Smart Cities"
        ]
    
    def get_category_prompt(self, category: str) -> str:
        prompts = {
            "Healthcare": "What are the current challenges in healthcare delivery, patient care, or medical technology?",
            "Education": "What problems exist in modern education systems, remote learning, or educational access?",
            "Climate Change": "What are pressing environmental challenges that need technological solutions?",
            "Transportation": "What mobility and transportation problems need solving in urban areas?",
            "Agriculture": "What challenges do farmers and the agricultural industry face?",
            "Business": "What inefficiencies exist in modern business operations?",
            "Logistics": "What supply chain and delivery challenges need solutions?",
            "Social Impact": "What social inequalities or community problems need addressing?",
            "Financial Inclusion": "What problems prevent access to financial services?",
            "Smart Cities": "What urban challenges could be solved with technology?"
        }
        return prompts.get(category, "What real-world problems need innovative solutions?")

class ThematicAnalyzer:
    """
    Analyzes themes and generates problem statements using AI model.
    Manages theme data, problem generation workflow, and database storage.
    """
    
    def __init__(self):
        self.themes = self._load_themes()
        self.crawler = WebCrawler(max_depth=1, max_pages=3)
        self.insight_cache = LRUCache(maxsize=100)
        self.db = DatabaseHandler()
        self.scraper = ProblemScraper()
        self.current_problems = []  # Track generated problems
        
    @staticmethod
    @lru_cache(maxsize=1)
    def _load_themes():
        """Cache themes data"""
        return {
            "AI & ML": {
                "description": "Artificial Intelligence and Machine Learning challenges",
                "keywords": ["AI ethics", "ML applications", "neural networks", "data science"],
                "sources": [
                    "https://ai.google/research/pubs/",
                    "https://openai.com/blog/",
                    "https://arxiv.org/list/cs.AI/recent",
                    "https://www.deepmind.com/blog"
                ]
            },
            "Climate Tech": {
                "description": "Climate change and environmental technology solutions",
                "keywords": ["renewable energy", "carbon capture", "sustainability"],
                "sources": [
                    "https://climatechange.mit.edu/",
                    "https://www.iea.org/topics/climate-change",
                    "https://www.climate.gov/"
                ]
            },
            "Healthcare": {
                "description": "Transforming healthcare through groundbreaking technologies",
                "keywords": ["patient care", "medical innovation", "healthcare operations"],
                "sources": [
                    "https://www.who.int/news-room/",
                    "https://www.healthcareitnews.com/",
                    "https://www.nature.com/subjects/health-care"
                ]
            },
            "Innovation": {
                "description": "Empowering bold ideas and creative solutions",
                "keywords": ["real-world challenges", "smart solutions", "breakthrough technology"],
                "sources": [
                    "https://www.wired.com/tag/innovation/",
                    "https://techcrunch.com/",
                    "https://www.technologyreview.com/"
                ]
            },
            "FinTech": {
                "description": "Pioneering the future of finance",
                "keywords": ["security", "transparency", "decentralized technologies"],
                "sources": [
                    "https://www.finextra.com/",
                    "https://www.coindesk.com/",
                    "https://www.fintechnews.org/"
                ]
            },
            "Logistics": {
                "description": "Reimagining movement of goods and services",
                "keywords": ["supply chains", "connectivity", "optimization"],
                "sources": [
                    "https://www.supplychaindigital.com/",
                    "https://www.logisticsmgmt.com/",
                    "https://www.freightwaves.com/"
                ]
            },
            "Sustainability": {
                "description": "Driving sustainable change with innovative technologies",
                "keywords": ["environmental", "renewable energy", "clean technology"],
                "sources": [
                    "https://www.greentechmedia.com/",
                    "https://www.sustainable-tech.org/",
                    "https://cleantechnica.com/"
                ]
            }
        }
    
    @lru_cache(maxsize=50)
    def get_theme_metadata(self, theme: str) -> dict:
        """Cached theme metadata access"""
        return self.themes.get(theme, {})

    async def analyze_theme(self, theme: str, model) -> str:
        """Enhanced theme analysis with problem tracking"""
        cache_key = f"{theme}_{datetime.now().strftime('%Y%m%d')}"
        
        if cache_key in self.insight_cache:
            return self.insight_cache[cache_key]
        
        # Replace crawling with targeted scraping
        scraped_problems = self.scraper.scrape_problems(theme)
        
        # Store new problems in database
        for problem in scraped_problems:
            self.db.add_problem({
                'title': problem['title'],
                'description': problem['description'],
                'theme': theme,
                'difficulty': problem.get('difficulty', 'Medium'),
                'source_url': problem['source_url']
            })
        
        # Generate prompt from scraped data
        prompt = self._generate_problem_prompt(theme, scraped_problems)
        
        if not prompt:
            return self._generate_fallback_prompt(theme)
            
        self.insight_cache[cache_key] = prompt
        return prompt

    def _generate_fallback_prompt(self, theme: str) -> str:
        """Generate fallback prompt when crawling fails"""
        theme_data = self.get_theme_metadata(theme)
        return f"""Based on {theme} focusing on:
        Description: {theme_data.get('description', '')}
        Keywords: {', '.join(theme_data.get('keywords', []))}
        // ...rest of the prompt template...
        """

    def _generate_problem_prompt(self, theme: str, problems: List[Dict]) -> str:
        """Generate a detailed prompt based on scraped problems with better analysis"""
        theme_data = self.get_theme_metadata(theme)
        
        # Group problems by difficulty
        problems_by_difficulty = {
            'Easy': [],
            'Medium': [],
            'Hard': []
        }
        for p in problems:
            problems_by_difficulty[p.get('difficulty', 'Medium')].append(p)

        # Extract key insights
        insights = []
        keywords = set()
        for problem in problems:
            # Extract key phrases (3-4 words)
            text = f"{problem['title']} {problem['description']}"
            phrases = re.findall(r'\b(\w+(?:\s+\w+){2,3})\b', text.lower())
            keywords.update(phrases[:3])  # Take top 3 phrases
            
            # Extract main challenge
            challenge = problem['description'].split('.')[0] if problem['description'] else ''
            if len(challenge) > 50:
                insights.append(challenge)

        # Format insights
        problem_summary = "\n\nCurrent Challenges:\n"
        for diff, probs in problems_by_difficulty.items():
            if probs:
                problem_summary += f"\n{diff} Level Challenges:\n"
                for p in probs[:3]:  # Top 3 per difficulty
                    problem_summary += f"- {p['title']}: {p['description'][:150]}...\n"

        # Add trending keywords
        keyword_summary = "\nTrending Topics:\n- " + "\n- ".join(list(keywords)[:10])

        # Combine everything into a structured prompt
        return f"""Analysis of {theme} Domain:
Description: {theme_data.get('description', '')}
Core Focus: {', '.join(theme_data.get('keywords', []))}

{problem_summary}

{keyword_summary}

Based on the above analysis, generate 5 specific, well-defined problem statements that:
1. Address real technical challenges in {theme}
2. Have clear success criteria
3. Can be prototyped in 2-4 weeks
4. Have measurable impact

For each problem, provide:

### [Problem Title]
**Challenge:** Technical description of the problem
**Context:** Why this matters now (reference current trends)
**Impact:** Quantifiable benefits and affected stakeholders
**Constraints:** Key technical/business limitations
**Success Criteria:** How to measure a solution's effectiveness
**Difficulty:** Easy/Medium/Hard (with brief explanation)

Requirements:
- Focus on practical, implementable solutions
- Consider current technology trends
- Address real stakeholder needs
- Enable quick prototyping
- Have clear evaluation metrics"""

    def _generate_single_problem_prompt(self, theme: str, problem: Dict) -> str:
        """Generate focused prompt for single problem"""
        theme_data = self.get_theme_metadata(theme)
        
        return f"""Transform this {theme} problem into a hackathon challenge:

SOURCE PROBLEM:
Title: {problem['title']}
Description: {problem['description'][:300]}...

REQUIREMENTS:
- Must be implementable in 2-4 weeks
- Should use {', '.join(theme_data.get('keywords', []))}
- Must have clear success metrics

Format the response as:

TITLE: [One line problem title]

CHALLENGE:
- [3-4 bullet points describing core technical challenge]

IMPLEMENTATION:
- [2-3 bullet points on key technical requirements]

METRICS:
- [2-3 bullet points on measuring success]

DIFFICULTY: [Easy/Medium/Hard] - [One line explanation]"""

    def _summarize_content(self, model, content: str, theme: str) -> str:
        """Use AI to summarize scraped content"""
        summary_prompt = f"""Analyze this {theme} content and provide a concise summary:

Content: {content[:3000]}...

Format the summary as:

KEY POINTS:
- [3-4 main points from the content]

MAIN CHALLENGES:
- [2-3 key problems/challenges identified]

TRENDS:
- [2-3 current trends]

Keep each bullet point under 50 words and focus on technical/implementation aspects."""

        output = model.create_completion(
            summary_prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=0.95,
            stop=["User:", "\n\n\n"],
            echo=False
        )
        
        return self._clean_ai_output(output['choices'][0]['text'])

    async def analyze_theme_iteratively(self, theme: str, model) -> List[Dict]:
        """Generate single focused problem with AI summarization"""
        scraped_problems = self.scraper.scrape_problems(theme)
        
        if not scraped_problems:
            print("No problems found, using default theme data")
            scraped_problems = [{
                'title': f"Generic {theme} Challenge",
                'description': self.get_theme_metadata(theme)['description'],
                'source_url': 'generated',
                'difficulty': 'Medium'
            }]

        # First, summarize all scraped content
        print("\n Analyzing scraped content...")
        combined_content = "\n\n".join([
            f"Title: {p['title']}\n{p['description']}"
            for p in scraped_problems[:3]  # Limit to top 3 for better focus
        ])
        
        content_summary = self._summarize_content(model, combined_content, theme)
        print("\n" + "="*80)
        print(" Content Analysis:")
        print("="*80)
        print(content_summary)
        print("="*80)

        # Generate problem based on summary
        problem = scraped_problems[0]
        prompt = self._generate_single_problem_prompt(
            theme, 
            {
                'title': problem['title'],
                'description': content_summary,  # Use AI summary instead of raw description
                'source_url': problem['source_url'],
                'difficulty': problem.get('difficulty', 'Medium')
            }
        )
        
        print("\nGenerating problem statement...")
        output = model.create_completion(
            prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=0.95,
            stop=["User:", "\n\n\n"],
            echo=False
        )
        
        problem_statement = self._clean_ai_output(output['choices'][0]['text'])
        
        print("\n" + "="*80)
        print(" Generated Problem:")
        print("="*80)
        print(problem_statement)
        print("="*80)
        
        # Store in database
        self.db.add_problem({
            'title': f"Generated Problem for {theme}",
            'description': problem_statement,
            'theme': theme,
            'source_url': problem['source_url'],
            'difficulty': 'Medium'
        })
        
        return [{
            'source': problem['source_url'],
            'original_title': problem['title'],
            'generated_content': problem_statement
        }]

    def _clean_ai_output(self, text: str) -> str:
        """Clean AI output by removing thinking process"""
        if '</think>' in text:
            # Take only the part after </think>
            parts = text.split('</think>')
            return parts[-1].strip()
        return text.strip()

    def _save_problem(self, theme: str, content: str, number: int):
        """Save problem to database instead of file"""
        self.db.add_problem({
            'title': f"Problem {number} for {theme}",
            'description': content,
            'theme': theme,
            'source_url': 'generated',
            'difficulty': 'Medium'
        })

async def main():
    """
    Main application loop that handles user interaction and problem generation.
    Coordinates between model loading, theme selection, and problem creation.
    """
    logger = setup_logging()
    analyzer = ThematicAnalyzer()
    
    model_path = "ai/DeepSeek-R1.gguf"
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at: {model_path}")
        return
    
    try:
        print("\nInitializing Problem Generator...")
        print("-" * 60)
        logger.info("Loading AI model...")
        model = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=4096,
            n_batch=512,
            verbose=False
        )
        print("AI model loaded successfully!")
        print("-" * 60)
        
        while True:
            print("\nAvailable Themes:")
            print("-" * 60)
            for i, theme in enumerate(analyzer.themes.keys(), 1):
                print(f"{i}. {theme}")
            print(f"{len(analyzer.themes) + 1}. Exit")
            
            try:
                choice = int(input("\nSelect theme number: "))
                if choice == len(analyzer.themes) + 1:
                    break
                    
                if 1 <= choice <= len(analyzer.themes):
                    theme = list(analyzer.themes.keys())[choice - 1]
                    print("\nProcess Flow:")
                    print("-" * 60)
                    
                    # Step 1: Scraping
                    print("[1/4] Gathering data from sources...")
                    problems = analyzer.scraper.scrape_problems(theme)
                    print(f"Found {len(problems)} relevant problems")
                    
                    # Step 2: Summarization
                    print("\n[2/4] Summarizing scraped data...")
                    combined_content = "\n\n".join([
                        f"Title: {p['title']}\n{p['description']}"
                        for p in problems[:3]
                    ])
                    content_summary = analyzer._summarize_content(model, combined_content, theme)
                    print("\nContent Analysis:")
                    print("-" * 60)
                    print(content_summary)
                    
                    # Step 3: Prompt Generation
                    print("\n[3/4] Generating problem prompt...")
                    prompt = analyzer._generate_single_problem_prompt(
                        theme,
                        {
                            'title': problems[0]['title'] if problems else f"Generic {theme} Challenge",
                            'description': content_summary,
                            'source_url': problems[0]['source_url'] if problems else 'generated',
                            'difficulty': 'Medium'
                        }
                    )
                    
                    # Step 4: Problem Generation
                    print("\n[4/4] Generating final problem statement...")
                    output = model.create_completion(
                        prompt,
                        max_tokens=512,
                        temperature=0.7,
                        top_p=0.95,
                        stop=["User:", "\n\n\n"],
                        echo=False
                    )
                    
                    problem_statement = analyzer._clean_ai_output(output['choices'][0]['text'])
                    
                    print("\nGenerated Problem Statement:")
                    print("-" * 60)
                    print(problem_statement)
                    
                    # Save to database
                    analyzer.db.add_problem({
                        'title': f"Generated Problem for {theme}",
                        'description': problem_statement,
                        'theme': theme,
                        'source_url': problems[0]['source_url'] if problems else 'generated',
                        'difficulty': 'Medium'
                    })
                    print("\nProblem saved to database successfully!")
                    
                else:
                    print("Invalid theme number!")
                    
            except ValueError:
                print("Please enter a valid number!")
            except Exception as e:
                logger.error(f"Error processing theme: {str(e)}")
                print("An error occurred. Please try another theme or check the logs.")
                
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise
    finally:
        await analyzer.crawler.close_session()

if __name__ == "__main__":
    asyncio.run(main())