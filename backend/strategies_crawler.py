import requests
import feedparser
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any
from openai import OpenAI
import os
from bs4 import BeautifulSoup
import time
from dataclasses import dataclass
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

@dataclass
class Strategy:
    title: str
    description: str
    category: str
    source: str
    url: str
    use_cases: List[str]
    code_examples: List[str]
    implementation_steps: List[str]
    ai_insights: str
    created_at: datetime

class StrategyEntry(Base):
    __tablename__ = 'strategies'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    category = Column(String(100), nullable=False)
    source = Column(String(200), nullable=False)
    url = Column(String(500), nullable=False)
    use_cases = Column(Text)  # JSON string
    code_examples = Column(Text)  # JSON string
    implementation_steps = Column(Text)  # JSON string
    ai_insights = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class StrategiesCrawler:
    def __init__(self, openai_api_key: str = None):
        self.openai_client = OpenAI(api_key=openai_api_key or os.getenv('OPENAI_API_KEY'))
        self.newsletter_sources = [
            {
                'name': 'AI News',
                'url': 'https://feeds.feedburner.com/artificial-intelligence-news',
                'category': 'AI/ML'
            },
            {
                'name': 'TechCrunch AI',
                'url': 'https://techcrunch.com/tag/artificial-intelligence/feed/',
                'category': 'AI/ML'
            },
            {
                'name': 'VentureBeat AI',
                'url': 'https://venturebeat.com/category/ai/feed/',
                'category': 'AI/ML'
            },
            {
                'name': 'MIT Technology Review',
                'url': 'https://www.technologyreview.com/topic/artificial-intelligence/feed',
                'category': 'AI/ML'
            }
        ]
        
        # Your knowledge base and vision
        self.knowledge_base = {
            'vision': """
            DataSafe aims to be the premier platform for AI strategy implementation, 
            combining cutting-edge AI research with practical business applications. 
            We focus on democratizing AI access for businesses of all sizes, 
            emphasizing ethical AI deployment and measurable ROI.
            """,
            'expertise_areas': [
                'AI Strategy Development',
                'Machine Learning Implementation',
                'Data Strategy',
                'AI Ethics and Governance',
                'Business Process Automation',
                'Customer Experience AI',
                'Predictive Analytics',
                'AI-Powered Marketing',
                'Supply Chain AI',
                'Financial AI Applications'
            ],
            'implementation_principles': [
                'Start with clear business objectives',
                'Ensure data quality and governance',
                'Build incrementally with quick wins',
                'Focus on user adoption and change management',
                'Measure and iterate continuously',
                'Maintain ethical AI practices',
                'Scale gradually with proven success'
            ]
        }

    def crawl_newsletters(self, days_back: int = 7) -> List[Dict[str, Any]]:
        """Crawl newsletter sources for recent content"""
        articles = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for source in self.newsletter_sources:
            try:
                print(f"Crawling {source['name']}...")
                feed = feedparser.parse(source['url'])
                
                for entry in feed.entries:
                    # Parse publication date
                    pub_date = datetime.now()
                    if hasattr(entry, 'published_parsed'):
                        pub_date = datetime(*entry.published_parsed[:6])
                    
                    if pub_date >= cutoff_date:
                        article = {
                            'title': entry.title,
                            'description': entry.get('summary', ''),
                            'url': entry.link,
                            'source': source['name'],
                            'category': source['category'],
                            'published_date': pub_date
                        }
                        articles.append(article)
                
                time.sleep(1)  # Be respectful to RSS feeds
                
            except Exception as e:
                print(f"Error crawling {source['name']}: {e}")
                continue
        
        return articles

    def extract_strategies_from_content(self, content: str, source: str) -> List[Strategy]:
        """Extract potential strategies from content using AI"""
        try:
            prompt = f"""
            Analyze the following content and extract actionable AI strategies, use cases, and implementation insights.
            
            Content: {content[:4000]}  # Limit content length
            
            Source: {source}
            
            Please identify and extract:
            1. AI strategies and approaches mentioned
            2. Specific use cases and applications
            3. Implementation steps or recommendations
            4. Code examples or technical details
            5. Business insights and ROI considerations
            
            Format your response as a JSON array of strategy objects with the following structure:
            {{
                "title": "Strategy title",
                "description": "Detailed description",
                "category": "AI/ML category",
                "use_cases": ["use case 1", "use case 2"],
                "implementation_steps": ["step 1", "step 2"],
                "code_examples": ["code snippet 1", "code snippet 2"],
                "ai_insights": "AI-generated insights and recommendations"
            }}
            
            Focus on practical, implementable strategies that businesses can use.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an AI strategy expert who extracts actionable insights from content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            strategies_data = json.loads(response.choices[0].message.content)
            strategies = []
            
            for strategy_data in strategies_data:
                strategy = Strategy(
                    title=strategy_data.get('title', ''),
                    description=strategy_data.get('description', ''),
                    category=strategy_data.get('category', 'AI/ML'),
                    source=source,
                    url='',
                    use_cases=strategy_data.get('use_cases', []),
                    code_examples=strategy_data.get('code_examples', []),
                    implementation_steps=strategy_data.get('implementation_steps', []),
                    ai_insights=strategy_data.get('ai_insights', ''),
                    created_at=datetime.now()
                )
                strategies.append(strategy)
            
            return strategies
            
        except Exception as e:
            print(f"Error extracting strategies: {e}")
            return []

    def enhance_with_knowledge_base(self, strategies: List[Strategy]) -> List[Strategy]:
        """Enhance strategies with your knowledge base and vision"""
        enhanced_strategies = []
        
        for strategy in strategies:
            try:
                enhancement_prompt = f"""
                Enhance this AI strategy with expert knowledge and implementation guidance.
                
                Original Strategy:
                Title: {strategy.title}
                Description: {strategy.description}
                Use Cases: {strategy.use_cases}
                
                Knowledge Base Context:
                Vision: {self.knowledge_base['vision']}
                Expertise Areas: {self.knowledge_base['expertise_areas']}
                Implementation Principles: {self.knowledge_base['implementation_principles']}
                
                Please enhance this strategy by:
                1. Adding more specific implementation steps
                2. Providing code examples where relevant
                3. Including business impact and ROI considerations
                4. Adding ethical AI considerations
                5. Suggesting measurement and evaluation approaches
                
                Return the enhanced strategy as JSON.
                """
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an AI strategy expert with deep implementation experience."},
                        {"role": "user", "content": enhancement_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=1500
                )
                
                enhanced_data = json.loads(response.choices[0].message.content)
                
                # Update strategy with enhanced content
                strategy.description = enhanced_data.get('description', strategy.description)
                strategy.use_cases.extend(enhanced_data.get('additional_use_cases', []))
                strategy.implementation_steps.extend(enhanced_data.get('additional_steps', []))
                strategy.code_examples.extend(enhanced_data.get('additional_code', []))
                strategy.ai_insights = enhanced_data.get('enhanced_insights', strategy.ai_insights)
                
                enhanced_strategies.append(strategy)
                
            except Exception as e:
                print(f"Error enhancing strategy: {e}")
                enhanced_strategies.append(strategy)
        
        return enhanced_strategies

    def generate_vision_aligned_strategies(self) -> List[Strategy]:
        """Generate new strategies based on your vision and expertise"""
        try:
            vision_prompt = f"""
            Based on the following vision and expertise, generate innovative AI strategies that businesses can implement.
            
            Vision: {self.knowledge_base['vision']}
            Expertise Areas: {self.knowledge_base['expertise_areas']}
            Implementation Principles: {self.knowledge_base['implementation_principles']}
            
            Generate 5 innovative AI strategies that:
            1. Align with the vision of democratizing AI access
            2. Focus on practical business applications
            3. Include specific implementation steps
            4. Provide code examples where relevant
            5. Consider ethical AI practices
            6. Include measurement and ROI approaches
            
            Format as JSON array of strategy objects.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an AI strategy innovator who creates cutting-edge, implementable strategies."},
                    {"role": "user", "content": vision_prompt}
                ],
                temperature=0.7,
                max_tokens=3000
            )
            
            strategies_data = json.loads(response.choices[0].message.content)
            strategies = []
            
            for strategy_data in strategies_data:
                strategy = Strategy(
                    title=strategy_data.get('title', ''),
                    description=strategy_data.get('description', ''),
                    category=strategy_data.get('category', 'AI Innovation'),
                    source='DataSafe Vision',
                    url='',
                    use_cases=strategy_data.get('use_cases', []),
                    code_examples=strategy_data.get('code_examples', []),
                    implementation_steps=strategy_data.get('implementation_steps', []),
                    ai_insights=strategy_data.get('ai_insights', ''),
                    created_at=datetime.now()
                )
                strategies.append(strategy)
            
            return strategies
            
        except Exception as e:
            print(f"Error generating vision-aligned strategies: {e}")
            return []

    def run_full_crawl(self) -> List[Strategy]:
        """Run the complete crawling and strategy generation process"""
        print("Starting full strategy crawl...")
        
        # 1. Crawl newsletters
        articles = self.crawl_newsletters(days_back=7)
        print(f"Found {len(articles)} recent articles")
        
        # 2. Extract strategies from articles
        all_strategies = []
        for article in articles:
            content = f"{article['title']}\n\n{article['description']}"
            strategies = self.extract_strategies_from_content(content, article['source'])
            all_strategies.extend(strategies)
        
        print(f"Extracted {len(all_strategies)} strategies from articles")
        
        # 3. Enhance with knowledge base
        enhanced_strategies = self.enhance_with_knowledge_base(all_strategies)
        print(f"Enhanced {len(enhanced_strategies)} strategies")
        
        # 4. Generate vision-aligned strategies
        vision_strategies = self.generate_vision_aligned_strategies()
        print(f"Generated {len(vision_strategies)} vision-aligned strategies")
        
        # 5. Combine all strategies
        all_strategies.extend(vision_strategies)
        
        print(f"Total strategies generated: {len(all_strategies)}")
        return all_strategies

    def save_strategies_to_database(self, strategies: List[Strategy], db_session):
        """Save strategies to the database"""
        for strategy in strategies:
            try:
                db_strategy = StrategyEntry(
                    title=strategy.title,
                    description=strategy.description,
                    category=strategy.category,
                    source=strategy.source,
                    url=strategy.url,
                    use_cases=json.dumps(strategy.use_cases),
                    code_examples=json.dumps(strategy.code_examples),
                    implementation_steps=json.dumps(strategy.implementation_steps),
                    ai_insights=strategy.ai_insights,
                    created_at=strategy.created_at
                )
                db_session.add(db_strategy)
            except Exception as e:
                print(f"Error saving strategy {strategy.title}: {e}")
                continue
        
        try:
            db_session.commit()
            print(f"Successfully saved {len(strategies)} strategies to database")
        except Exception as e:
            print(f"Error committing strategies to database: {e}")
            db_session.rollback()

# Example usage
if __name__ == "__main__":
    crawler = StrategiesCrawler()
    strategies = crawler.run_full_crawl()
    
    # Print sample strategies
    for i, strategy in enumerate(strategies[:3]):
        print(f"\n--- Strategy {i+1} ---")
        print(f"Title: {strategy.title}")
        print(f"Category: {strategy.category}")
        print(f"Source: {strategy.source}")
        print(f"Use Cases: {strategy.use_cases}")
        print(f"Implementation Steps: {len(strategy.implementation_steps)} steps")
        print(f"Code Examples: {len(strategy.code_examples)} examples")
        print(f"AI Insights: {strategy.ai_insights[:200]}...") 