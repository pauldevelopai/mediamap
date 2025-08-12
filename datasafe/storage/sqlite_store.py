"""
SQLite storage for normalized threat intelligence data
"""
import sqlite3
import json
import os
from typing import Dict, List, Optional
from datetime import datetime

DB_PATH = os.getenv('DATASAFE_DB_PATH', 'datasafe.db')

def ensure_schema():
    """Create the database schema if it doesn't exist"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS threats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT NOT NULL,
        title TEXT NOT NULL,
        summary TEXT,
        threats TEXT,  -- JSON array
        sectors TEXT,  -- JSON array
        iocs TEXT,     -- JSON object
        url TEXT,
        published_at TEXT,
        severity TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create indexes for common queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON threats(source)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_severity ON threats(severity)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_published_at ON threats(published_at)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON threats(created_at)')
    
    conn.commit()
    conn.close()

def insert_record(record: Dict) -> bool:
    """Insert a normalized threat record into the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
        INSERT INTO threats (
            source, title, summary, threats, sectors, iocs, 
            url, published_at, severity
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            record.get('source', ''),
            record.get('title', ''),
            record.get('summary', ''),
            json.dumps(record.get('threats', [])),
            json.dumps(record.get('sectors', [])),
            json.dumps(record.get('iocs', {})),
            record.get('url'),
            record.get('published_at'),
            record.get('severity', 'Low')
        ))
        
        conn.commit()
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to insert record: {e}")
        return False
    finally:
        conn.close()

def query_threats(
    source: Optional[str] = None,
    severity: Optional[str] = None,
    sector: Optional[str] = None,
    limit: int = 100
) -> List[Dict]:
    """Query threats with optional filters"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    query = "SELECT * FROM threats WHERE 1=1"
    params = []
    
    if source:
        query += " AND source = ?"
        params.append(source)
    
    if severity:
        query += " AND severity = ?"
        params.append(severity)
    
    if sector:
        query += " AND sectors LIKE ?"
        params.append(f'%"{sector}"%')
    
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    # Convert to dictionaries
    columns = [desc[0] for desc in cursor.description]
    results = []
    
    for row in rows:
        record = dict(zip(columns, row))
        # Parse JSON fields
        try:
            record['threats'] = json.loads(record['threats'] or '[]')
            record['sectors'] = json.loads(record['sectors'] or '[]')
            record['iocs'] = json.loads(record['iocs'] or '{}')
        except json.JSONDecodeError:
            record['threats'] = []
            record['sectors'] = []
            record['iocs'] = {}
        
        results.append(record)
    
    conn.close()
    return results

def get_stats() -> Dict:
    """Get database statistics"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    stats = {}
    
    # Total records
    cursor.execute("SELECT COUNT(*) FROM threats")
    stats['total_records'] = cursor.fetchone()[0]
    
    # Records by source
    cursor.execute("SELECT source, COUNT(*) FROM threats GROUP BY source")
    stats['by_source'] = dict(cursor.fetchall())
    
    # Records by severity
    cursor.execute("SELECT severity, COUNT(*) FROM threats GROUP BY severity")
    stats['by_severity'] = dict(cursor.fetchall())
    
    # Latest record
    cursor.execute("SELECT MAX(created_at) FROM threats")
    latest = cursor.fetchone()[0]
    stats['latest_record'] = latest
    
    conn.close()
    return stats

def clear_database():
    """Clear all records from the database (for testing)"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM threats")
    conn.commit()
    conn.close()
