"""
Enhanced Data Collection System for Custom AI Model Training

This module extends the basic data collector with internet-based data sources:
1. arXiv research papers
2. Industry blogs and whitepapers
3. Public AI/ML datasets
4. News articles and reports
5. Technical documentation
6. GitHub repositories and code examples
"""

import os
import json
import requests
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import feedparser
from bs4 import BeautifulSoup
import hashlib
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDataCollector:
    """Enhanced data collection system with internet sources"""
    
    def __init__(self, output_dir: str = "./training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different data types
        (self.output_dir / "internet_sources").mkdir(exist_ok=True)
        (self.output_dir / "arxiv_papers").mkdir(exist_ok=True)
        (self.output_dir / "industry_content").mkdir(exist_ok=True)
        (self.output_dir / "public_datasets").mkdir(exist_ok=True)
        (self.output_dir / "news_articles").mkdir(exist_ok=True)
        (self.output_dir / "technical_docs").mkdir(exist_ok=True)
        
        # Define the review data file path
        self.review_data_file = self.output_dir / "internet_sources" / "detailed_review_data.json"
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
        
        logger.info("EnhancedDataCollector initialized")
    
    def _rate_limit(self):
        """Simple rate limiting to be respectful to APIs"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def collect_arxiv_papers(self, max_papers: int = 50) -> int:
        """Collect recent AI/ML papers from arXiv"""
        logger.info("Collecting arXiv papers...")
        
        try:
            # arXiv API endpoint
            base_url = "http://export.arxiv.org/api/query"
            
            # Search for recent AI/ML papers
            search_queries = [
                "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.NE",
                "cat:stat.ML OR cat:cs.CV OR cat:cs.RO",
                "cat:cs.CR OR cat:cs.CY OR cat:cs.HC"
            ]
            
            papers_collected = 0
            
            for query in search_queries:
                if papers_collected >= max_papers:
                    break
                    
                params = {
                    'search_query': query,
                    'start': 0,
                    'max_results': min(20, max_papers - papers_collected),
                    'sortBy': 'submittedDate',
                    'sortOrder': 'descending'
                }
                
                self._rate_limit()
                response = requests.get(base_url, params=params, timeout=30)
                
                if response.status_code == 200:
                    # Parse XML response
                    soup = BeautifulSoup(response.content, 'xml')
                    entries = soup.find_all('entry')
                    
                    for entry in entries:
                        if papers_collected >= max_papers:
                            break
                            
                        paper_data = {
                            'id': entry.find('id').text if entry.find('id') else '',
                            'title': entry.find('title').text.strip() if entry.find('title') else '',
                            'summary': entry.find('summary').text.strip() if entry.find('summary') else '',
                            'authors': [author.find('name').text for author in entry.find_all('author')],
                            'published': entry.find('published').text if entry.find('published') else '',
                            'updated': entry.find('updated').text if entry.find('updated') else '',
                            'categories': [cat.get('term') for cat in entry.find_all('category')],
                            'links': [link.get('href') for link in entry.find_all('link')],
                            'source': 'arxiv'
                        }
                        
                        # Save individual paper
                        paper_id = paper_data['id'].split('/')[-1] if paper_data['id'] else f"paper_{papers_collected}"
                        output_file = self.output_dir / "arxiv_papers" / f"{paper_id}.json"
                        
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(paper_data, f, indent=2, ensure_ascii=False)
                        
                        papers_collected += 1
                        logger.info(f"Collected arXiv paper: {paper_data['title'][:50]}...")
                
                time.sleep(2)  # Be respectful to arXiv API
            
            logger.info(f"Collected {papers_collected} arXiv papers")
            return papers_collected
            
        except Exception as e:
            logger.error(f"Error collecting arXiv papers: {e}")
            return 0
    
    def collect_industry_content(self) -> int:
        """Collect industry blogs, whitepapers, and technical content"""
        logger.info("Collecting industry content...")
        
        # Industry sources and their RSS feeds or APIs
        sources = [
            {
                'name': 'OpenAI Blog',
                'url': 'https://openai.com/blog/rss.xml',
                'type': 'rss'
            },
            {
                'name': 'Google AI Blog',
                'url': 'https://ai.googleblog.com/feeds/posts/default',
                'type': 'rss'
            },
            {
                'name': 'Anthropic Blog',
                'url': 'https://www.anthropic.com/news/rss',
                'type': 'rss'
            },
            {
                'name': 'Hugging Face Blog',
                'url': 'https://huggingface.co/blog/rss.xml',
                'type': 'rss'
            }
        ]
        
        content_collected = 0
        
        for source in sources:
            try:
                if source['type'] == 'rss':
                    self._rate_limit()
                    feed = feedparser.parse(source['url'])
                    
                    for entry in feed.entries[:10]:  # Limit to 10 most recent
                        content_data = {
                            'title': entry.get('title', ''),
                            'summary': entry.get('summary', ''),
                            'link': entry.get('link', ''),
                            'published': entry.get('published', ''),
                            'source': source['name'],
                            'type': 'blog_post'
                        }
                        
                        # Create filename from title
                        safe_title = re.sub(r'[^\w\s-]', '', content_data['title'])
                        safe_title = re.sub(r'[-\s]+', '-', safe_title)[:50]
                        filename = f"{source['name'].replace(' ', '_')}_{safe_title}.json"
                        
                        output_file = self.output_dir / "industry_content" / filename
                        
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(content_data, f, indent=2, ensure_ascii=False)
                        
                        content_collected += 1
                        logger.info(f"Collected from {source['name']}: {content_data['title'][:50]}...")
                
                time.sleep(2)  # Be respectful to sources
                
            except Exception as e:
                logger.error(f"Error collecting from {source['name']}: {e}")
                continue
        
        logger.info(f"Collected {content_collected} industry content items")
        return content_collected
    
    def collect_public_datasets(self) -> int:
        """Collect information about public AI/ML datasets"""
        logger.info("Collecting public dataset information...")
        
        # Popular AI/ML datasets and their descriptions
        datasets = [
            {
                'name': 'Common Crawl',
                'description': 'Large-scale web crawl data for training language models',
                'url': 'https://commoncrawl.org/',
                'size': 'Petabytes',
                'type': 'web_crawl',
                'use_case': 'language_model_training'
            },
            {
                'name': 'The Pile',
                'description': 'Diverse text dataset for training large language models',
                'url': 'https://pile.eleuther.ai/',
                'size': '825GB',
                'type': 'text_corpus',
                'use_case': 'language_model_training'
            },
            {
                'name': 'C4 (Colossal Clean Crawled Corpus)',
                'description': 'Cleaned version of Common Crawl for training T5 and other models',
                'url': 'https://www.tensorflow.org/datasets/catalog/c4',
                'size': '750GB',
                'type': 'text_corpus',
                'use_case': 'language_model_training'
            },
            {
                'name': 'OpenWebText',
                'description': 'Open source recreation of WebText dataset',
                'url': 'https://github.com/jcpeterson/openwebtext',
                'size': '38GB',
                'type': 'text_corpus',
                'use_case': 'language_model_training'
            },
            {
                'name': 'Wikipedia',
                'description': 'Complete Wikipedia dump for training and fine-tuning',
                'url': 'https://dumps.wikimedia.org/',
                'size': '20GB+',
                'type': 'encyclopedia',
                'use_case': 'knowledge_training'
            },
            {
                'name': 'Stack Overflow',
                'description': 'Programming Q&A data for code generation training',
                'url': 'https://archive.org/details/stackexchange',
                'size': '50GB+',
                'type': 'qa_data',
                'use_case': 'code_generation'
            },
            {
                'name': 'Reddit Comments',
                'description': 'Large collection of Reddit comments for conversational AI',
                'url': 'https://files.pushshift.io/reddit/comments/',
                'size': '1TB+',
                'type': 'conversational',
                'use_case': 'dialogue_training'
            },
            {
                'name': 'Books3',
                'description': 'Collection of books for training language models',
                'url': 'https://the-eye.eu/public/AI/pile_preliminary_components/',
                'size': '37GB',
                'type': 'literature',
                'use_case': 'literary_training'
            }
        ]
        
        datasets_collected = 0
        
        for dataset in datasets:
            dataset_info = {
                'name': dataset['name'],
                'description': dataset['description'],
                'url': dataset['url'],
                'size': dataset['size'],
                'type': dataset['type'],
                'use_case': dataset['use_case'],
                'collected_at': datetime.now().isoformat(),
                'source': 'public_datasets'
            }
            
            # Create filename from dataset name
            safe_name = re.sub(r'[^\w\s-]', '', dataset['name'])
            safe_name = re.sub(r'[-\s]+', '_', safe_name)
            filename = f"{safe_name}.json"
            
            output_file = self.output_dir / "public_datasets" / filename
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset_info, f, indent=2, ensure_ascii=False)
            
            datasets_collected += 1
            logger.info(f"Collected dataset info: {dataset['name']}")
        
        logger.info(f"Collected {datasets_collected} public dataset information")
        return datasets_collected
    
    def collect_news_articles(self) -> int:
        """Collect recent AI/ML news articles"""
        logger.info("Collecting AI/ML news articles...")
        
        # News sources with AI/ML focus
        news_sources = [
            {
                'name': 'AI News',
                'url': 'https://www.artificialintelligence-news.com/feed/',
                'type': 'rss'
            },
            {
                'name': 'MIT Technology Review AI',
                'url': 'https://www.technologyreview.com/topic/artificial-intelligence/feed/',
                'type': 'rss'
            },
            {
                'name': 'VentureBeat AI',
                'url': 'https://venturebeat.com/ai/feed/',
                'type': 'rss'
            }
        ]
        
        articles_collected = 0
        
        for source in news_sources:
            try:
                if source['type'] == 'rss':
                    self._rate_limit()
                    feed = feedparser.parse(source['url'])
                    
                    for entry in feed.entries[:5]:  # Limit to 5 most recent
                        article_data = {
                            'title': entry.get('title', ''),
                            'summary': entry.get('summary', ''),
                            'link': entry.get('link', ''),
                            'published': entry.get('published', ''),
                            'source': source['name'],
                            'type': 'news_article'
                        }
                        
                        # Create filename from title
                        safe_title = re.sub(r'[^\w\s-]', '', article_data['title'])
                        safe_title = re.sub(r'[-\s]+', '-', safe_title)[:50]
                        filename = f"{source['name'].replace(' ', '_')}_{safe_title}.json"
                        
                        output_file = self.output_dir / "news_articles" / filename
                        
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(article_data, f, indent=2, ensure_ascii=False)
                        
                        articles_collected += 1
                        logger.info(f"Collected news from {source['name']}: {article_data['title'][:50]}...")
                
                time.sleep(2)  # Be respectful to sources
                
            except Exception as e:
                logger.error(f"Error collecting news from {source['name']}: {e}")
                continue
        
        logger.info(f"Collected {articles_collected} news articles")
        return articles_collected
    
    def collect_technical_documentation(self) -> int:
        """Collect technical documentation and guides"""
        logger.info("Collecting technical documentation...")
        
        # Technical documentation sources
        docs_sources = [
            {
                'name': 'PyTorch Documentation',
                'url': 'https://pytorch.org/docs/stable/',
                'type': 'documentation',
                'topics': ['deep_learning', 'neural_networks', 'pytorch']
            },
            {
                'name': 'TensorFlow Documentation',
                'url': 'https://www.tensorflow.org/api_docs',
                'type': 'documentation',
                'topics': ['deep_learning', 'neural_networks', 'tensorflow']
            },
            {
                'name': 'Hugging Face Documentation',
                'url': 'https://huggingface.co/docs',
                'type': 'documentation',
                'topics': ['transformers', 'nlp', 'huggingface']
            },
            {
                'name': 'OpenAI API Documentation',
                'url': 'https://platform.openai.com/docs',
                'type': 'documentation',
                'topics': ['api', 'gpt', 'openai']
            }
        ]
        
        docs_collected = 0
        
        for source in docs_sources:
            doc_info = {
                'name': source['name'],
                'url': source['url'],
                'type': source['type'],
                'topics': source['topics'],
                'collected_at': datetime.now().isoformat(),
                'source': 'technical_docs',
                'note': 'Documentation metadata collected - full content extraction would require web scraping'
            }
            
            # Create filename from source name
            safe_name = re.sub(r'[^\w\s-]', '', source['name'])
            safe_name = re.sub(r'[-\s]+', '_', safe_name)
            filename = f"{safe_name}.json"
            
            output_file = self.output_dir / "technical_docs" / filename
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(doc_info, f, indent=2, ensure_ascii=False)
            
            docs_collected += 1
            logger.info(f"Collected documentation info: {source['name']}")
        
        logger.info(f"Collected {docs_collected} technical documentation sources")
        return docs_collected
    
    def collect_all_internet_sources(self) -> Dict[str, int]:
        """Collect data from all internet sources"""
        logger.info("Starting comprehensive internet data collection...")
        
        stats = {
            'arxiv_papers': self.collect_arxiv_papers(max_papers=30),
            'industry_content': self.collect_industry_content(),
            'public_datasets': self.collect_public_datasets(),
            'news_articles': self.collect_news_articles(),
            'technical_docs': self.collect_technical_documentation()
        }
        
        total_items = sum(stats.values())
        stats['total_internet_items'] = total_items
        
        logger.info(f"Internet data collection complete: {stats}")
        return stats

    def collect_detailed_internet_sources(self) -> Dict[str, Any]:
        """Collect detailed data from internet sources for review and approval"""
        logger.info("Starting detailed internet data collection for review...")
        
        detailed_data = {
            'arxiv_papers': [],
            'industry_content': [],
            'public_datasets': [],
            'news_articles': [],
            'technical_docs': [],
            'total_items': 0,
            'collection_timestamp': datetime.now().isoformat()
        }
        
        try:
            # Collect detailed arXiv papers
            arxiv_items = self._collect_detailed_arxiv_papers()
            detailed_data['arxiv_papers'] = arxiv_items
            
            # Collect detailed industry content
            industry_items = self._collect_detailed_industry_content()
            detailed_data['industry_content'] = industry_items
            
            # Collect detailed public datasets
            dataset_items = self._collect_detailed_public_datasets()
            detailed_data['public_datasets'] = dataset_items
            
            # Collect detailed news articles
            news_items = self._collect_detailed_news_articles()
            detailed_data['news_articles'] = news_items
            
            # Collect detailed technical documentation
            docs_items = self._collect_detailed_technical_docs()
            detailed_data['technical_docs'] = docs_items
            
            # Calculate total items
            detailed_data['total_items'] = sum([
                len(detailed_data['arxiv_papers']),
                len(detailed_data['industry_content']),
                len(detailed_data['public_datasets']),
                len(detailed_data['news_articles']),
                len(detailed_data['technical_docs'])
            ])
            
            # Save detailed data for review
            try:
                with open(self.review_data_file, 'w', encoding='utf-8') as f:
                    json.dump(detailed_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Detailed internet data collection complete: {detailed_data['total_items']} items saved to {self.review_data_file}")
            except Exception as save_error:
                logger.error(f"Error saving review data: {save_error}")
                detailed_data['save_error'] = str(save_error)
            
        except Exception as e:
            logger.error(f"Error in detailed internet data collection: {e}")
            detailed_data['error'] = str(e)
        
        return detailed_data

    def load_existing_review_data(self) -> Dict[str, Any]:
        """Load existing review data if it exists"""
        try:
            if self.review_data_file.exists():
                with open(self.review_data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"Loaded existing review data: {data.get('total_items', 0)} items")
                return data
            else:
                logger.info("No existing review data found")
                return None
        except Exception as e:
            logger.error(f"Error loading existing review data: {e}")
            return None

    def has_existing_data(self) -> bool:
        """Check if review data already exists"""
        return self.review_data_file.exists()
    
    def create_training_recommendations(self) -> Dict[str, Any]:
        """Create recommendations for training data usage"""
        recommendations = {
            'high_quality_sources': [
                'arXiv papers for technical knowledge',
                'Industry blogs for practical applications',
                'Public datasets for large-scale training',
                'Technical documentation for implementation details'
            ],
            'data_processing_suggestions': [
                'Clean and deduplicate collected data',
                'Extract key insights and summaries',
                'Create question-answer pairs from content',
                'Generate training examples from real-world scenarios'
            ],
            'model_training_advice': [
                'Start with high-quality, curated datasets',
                'Use domain-specific data for fine-tuning',
                'Implement data augmentation techniques',
                'Monitor training progress and adjust accordingly'
            ],
            'ethical_considerations': [
                'Respect robots.txt and terms of service',
                'Implement proper attribution for sources',
                'Ensure data privacy and security',
                'Follow fair use guidelines'
            ]
        }
        
        # Save recommendations
        output_file = self.output_dir / "internet_sources" / "training_recommendations.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, indent=2, ensure_ascii=False)
        
        return recommendations

    def _collect_detailed_arxiv_papers(self, max_papers: int = 20) -> List[Dict[str, Any]]:
        """Collect detailed arXiv papers for review"""
        papers = []
        try:
            # Search for AI/ML related papers
            query = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:stat.ML"
            url = f"http://export.arxiv.org/api/query?search_query={query}&start=0&max_results={max_papers}&sortBy=submittedDate&sortOrder=descending"
            
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'xml')
                entries = soup.find_all('entry')
                
                for entry in entries:
                    paper = {
                        'id': entry.find('id').text if entry.find('id') else '',
                        'title': entry.find('title').text.strip() if entry.find('title') else '',
                        'summary': entry.find('summary').text.strip() if entry.find('summary') else '',
                        'authors': [author.find('name').text if author.find('name') else '' for author in entry.find_all('author')],
                        'published': entry.find('published').text if entry.find('published') else '',
                        'updated': entry.find('updated').text if entry.find('updated') else '',
                        'categories': [cat.get('term') for cat in entry.find_all('category')],
                        'pdf_url': entry.find('link', {'type': 'application/pdf'}).get('href') if entry.find('link', {'type': 'application/pdf'}) else '',
                        'source': 'arXiv',
                        'type': 'research_paper',
                        'approved': False,
                        'review_notes': ''
                    }
                    papers.append(paper)
                    
        except Exception as e:
            logger.error(f"Error collecting detailed arXiv papers: {e}")
            
        return papers

    def _collect_detailed_industry_content(self) -> List[Dict[str, Any]]:
        """Collect detailed industry content for review"""
        content = []
        try:
            # Industry blog sources
            sources = [
                {
                    'name': 'OpenAI Blog',
                    'url': 'https://openai.com/blog/',
                    'type': 'blog_post'
                },
                {
                    'name': 'Google AI Blog',
                    'url': 'https://ai.googleblog.com/',
                    'type': 'blog_post'
                },
                {
                    'name': 'Microsoft AI Blog',
                    'url': 'https://blogs.microsoft.com/ai/',
                    'type': 'blog_post'
                }
            ]
            
            for source in sources:
                try:
                    response = requests.get(source['url'], timeout=30)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Find article links (this is a simplified example)
                        articles = soup.find_all('a', href=True)[:5]  # Limit to 5 articles per source
                        
                        for article in articles:
                            if article.get_text().strip():
                                item = {
                                    'title': article.get_text().strip(),
                                    'url': article['href'],
                                    'source': source['name'],
                                    'type': source['type'],
                                    'summary': f"Article from {source['name']}",
                                    'published': datetime.now().isoformat(),
                                    'approved': False,
                                    'review_notes': ''
                                }
                                content.append(item)
                                
                except Exception as e:
                    logger.error(f"Error collecting from {source['name']}: {e}")
                    
        except Exception as e:
            logger.error(f"Error collecting detailed industry content: {e}")
            
        return content

    def _collect_detailed_public_datasets(self) -> List[Dict[str, Any]]:
        """Collect detailed public datasets for review"""
        datasets = []
        try:
            # Popular AI/ML datasets
            dataset_info = [
                {
                    'name': 'ImageNet',
                    'description': 'Large-scale image database for visual recognition',
                    'size': '150GB+',
                    'type': 'image_classification',
                    'url': 'https://www.image-net.org/',
                    'source': 'Stanford Vision Lab'
                },
                {
                    'name': 'COCO',
                    'description': 'Common Objects in Context dataset',
                    'size': '20GB+',
                    'type': 'object_detection',
                    'url': 'https://cocodataset.org/',
                    'source': 'Microsoft'
                },
                {
                    'name': 'GLUE Benchmark',
                    'description': 'General Language Understanding Evaluation benchmark',
                    'size': '1GB+',
                    'type': 'natural_language_processing',
                    'url': 'https://gluebenchmark.com/',
                    'source': 'NYU'
                }
            ]
            
            for dataset in dataset_info:
                item = {
                    'name': dataset['name'],
                    'description': dataset['description'],
                    'size': dataset['size'],
                    'type': dataset['type'],
                    'url': dataset['url'],
                    'source': dataset['source'],
                    'category': 'public_dataset',
                    'approved': False,
                    'review_notes': ''
                }
                datasets.append(item)
                
        except Exception as e:
            logger.error(f"Error collecting detailed public datasets: {e}")
            
        return datasets

    def _collect_detailed_news_articles(self) -> List[Dict[str, Any]]:
        """Collect detailed news articles for review"""
        articles = []
        try:
            # News sources with RSS feeds
            news_sources = [
                {
                    'name': 'AI News',
                    'url': 'https://www.artificialintelligence-news.com/feed/',
                    'type': 'news_article'
                }
            ]
            
            for source in news_sources:
                try:
                    feed = feedparser.parse(source['url'])
                    for entry in feed.entries[:10]:  # Limit to 10 articles per source
                        article = {
                            'title': entry.get('title', ''),
                            'summary': entry.get('summary', ''),
                            'url': entry.get('link', ''),
                            'published': entry.get('published', ''),
                            'source': source['name'],
                            'type': source['type'],
                            'approved': False,
                            'review_notes': ''
                        }
                        articles.append(article)
                        
                except Exception as e:
                    logger.error(f"Error collecting from {source['name']}: {e}")
                    
        except Exception as e:
            logger.error(f"Error collecting detailed news articles: {e}")
            
        return articles

    def _collect_detailed_technical_docs(self) -> List[Dict[str, Any]]:
        """Collect detailed technical documentation for review"""
        docs = []
        try:
            # Technical documentation sources
            doc_sources = [
                {
                    'name': 'PyTorch Documentation',
                    'url': 'https://pytorch.org/docs/stable/',
                    'type': 'technical_documentation',
                    'description': 'PyTorch deep learning framework documentation'
                },
                {
                    'name': 'TensorFlow Documentation',
                    'url': 'https://www.tensorflow.org/api_docs',
                    'type': 'technical_documentation',
                    'description': 'TensorFlow machine learning platform documentation'
                },
                {
                    'name': 'Hugging Face Documentation',
                    'url': 'https://huggingface.co/docs',
                    'type': 'technical_documentation',
                    'description': 'Hugging Face transformers library documentation'
                }
            ]
            
            for source in doc_sources:
                doc = {
                    'name': source['name'],
                    'url': source['url'],
                    'type': source['type'],
                    'description': source['description'],
                    'source': 'Official Documentation',
                    'category': 'technical_documentation',
                    'approved': False,
                    'review_notes': ''
                }
                docs.append(doc)
                
        except Exception as e:
            logger.error(f"Error collecting detailed technical docs: {e}")
            
        return docs

if __name__ == "__main__":
    # Example usage
    collector = EnhancedDataCollector()
    stats = collector.collect_all_internet_sources()
    recommendations = collector.create_training_recommendations()
    
    print(f"Collection complete: {stats}")
    print(f"Recommendations: {recommendations}")
