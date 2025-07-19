# database.py
import sqlite3
from datetime import datetime

def init_database():
    """Initialize SQLite database for storing articles."""
    conn = sqlite3.connect('news_articles.db')
    cursor = conn.cursor()
    
    # Create articles table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT UNIQUE,
        title TEXT,
        content TEXT,
        summary TEXT,
        publish_date TIMESTAMP,
        source TEXT,
        relevance_score REAL,
        topic TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()

def store_article(article_data):
    """Store article in SQLite database."""
    conn = sqlite3.connect('news_articles.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            '''
            INSERT INTO articles (url, title, content, summary, publish_date, source, relevance_score, topic)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                article_data.get('url'),
                article_data.get('title'),
                article_data.get('content'),
                article_data.get('summary'),
                article_data.get('publish_date'),
                article_data.get('source'),
                article_data.get('relevance_score', 0.0),
                article_data.get('topic')
            )
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Article already exists
        return False
    finally:
        conn.close()

def get_sent_articles(topic, hours=24):
    """Get articles that have already been sent."""
    conn = sqlite3.connect('news_articles.db')
    cursor = conn.cursor()
    
    cursor.execute(
        '''
        SELECT url FROM articles 
        WHERE topic = ? AND 
        created_at > datetime('now', '-' || ? || ' hour')
        ''',
        (topic, hours)
    )
    
    urls = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return urls