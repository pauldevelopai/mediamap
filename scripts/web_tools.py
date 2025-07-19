# web_tools.py
import requests
import json
from newspaper import Article
from datetime import datetime
import sqlite3
from config import GOOGLE_API_KEY, SEARCH_ENGINE_ID

def web_search(query, num_results=10):
    """Search the web for news articles matching the query."""
    # Add recent news filter to query
    if "date" not in query.lower():
        query += " date:d"  # Restrict to past day
    
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query,
        "num": num_results,
        "sort": "date"
    }
    
    response = requests.get(url, params=params)
    results = []
    
    if response.status_code == 200:
        data = response.json()
        if "items" in data:
            for item in data["items"]:
                results.append({
                    "url": item.get("link"),
                    "title": item.get("title"),
                    "snippet": item.get("snippet"),
                    "source": item.get("displayLink"),
                    "date": item.get("pagemap", {}).get("metatags", [{}])[0].get("article:published_time")
                })
    
    return results

def extract_article_content(url):
    """Extract article content using newspaper3k library."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        # Try to extract publish date from article
        publish_date = article.publish_date
        if not publish_date:
            # Fallback: try to estimate from article
            publish_date = datetime.now()
        
        return {
            "url": url,
            "title": article.title,
            "content": article.text,
            "summary": article.summary,
            "publish_date": publish_date,
            "source": article.source_url or url.split("//")[1].split("/")[0]
        }
    except Exception as e:
        print(f"Error extracting content from {url}: {str(e)}")
        return None