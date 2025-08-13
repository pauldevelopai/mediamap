import os, json, sqlite3
from ..config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS threats(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT DEFAULT (datetime('now')),
  source TEXT, title TEXT, summary TEXT,
  threats TEXT, sectors TEXT, iocs TEXT,
  url TEXT, published_at TEXT, severity TEXT,
  actions TEXT, risk TEXT
);
CREATE INDEX IF NOT EXISTS ix_pub ON threats(published_at);
CREATE INDEX IF NOT EXISTS ix_sev ON threats(severity);
"""

def _cx():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True) if os.path.dirname(DB_PATH) else None
    cx=sqlite3.connect(DB_PATH)
    cx.execute("PRAGMA journal_mode=WAL;")
    return cx

def ensure_schema():
    with _cx() as c:
        for stmt in SCHEMA.strip().split(";"):
            if stmt.strip(): c.execute(stmt)
        c.commit()

def insert_record(rec:dict):
    with _cx() as c:
        c.execute("""INSERT INTO threats(source,title,summary,threats,sectors,iocs,url,published_at,severity,actions,risk)
                     VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                  (rec["source"], rec["title"], rec["summary"],
                   json.dumps(rec["threats"]), json.dumps(rec["sectors"]),
                   json.dumps(rec["iocs"]), rec["url"], rec["published_at"], rec["severity"],
                   json.dumps(rec["actions"]), json.dumps(rec["risk"])) )
        c.commit()

def query_recent(limit=200, min_risk=0, severity=None):
    q="SELECT ts,source,title,summary,threats,sectors,iocs,url,published_at,severity,actions,risk FROM threats"
    args=[]; where=[]
    if severity: where.append("severity=?"); args.append(severity)
    if min_risk>0: where.append("CAST(json_extract(risk,'$.digital') AS INT)>=?"); args.append(min_risk)
    if where: q+=" WHERE "+ " AND ".join(where)
    q+=" ORDER BY COALESCE(published_at, ts) DESC LIMIT ?"; args.append(limit)
    with _cx() as c:
        rows=c.execute(q,args).fetchall()
    out=[]
    for ts,src,title,sumry,thr,secs,iocs,url,pub,sev,acts,risk in rows:
        out.append({"ts":ts,"source":src,"title":title,"summary":sumry,
                    "threats":json.loads(thr),"sectors":json.loads(secs),
                    "iocs":json.loads(iocs),"url":url,"published_at":pub,"severity":sev,
                    "actions":json.loads(acts),"risk":json.loads(risk)})
    return out
"""
SQLite storage for normalized threat intelligence data
"""
import sqlite3
import json
import os
from typing import Dict, List, Optional
from datetime import datetime

DB_PATH = os.getenv('DATASAFE_DB', os.getenv('DATASAFE_DB_PATH', 'datasafe.db'))

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
        risk TEXT,     -- JSON object {digital, physical, explanations}
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create indexes for common queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON threats(source)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_severity ON threats(severity)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_published_at ON threats(published_at)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON threats(created_at)')
    
    conn.commit()

    # Backfill: add risk column if missing (SQLite: try/catch on ALTER)
    try:
        cursor.execute("SELECT risk FROM threats LIMIT 1")
    except sqlite3.OperationalError:
        try:
            cursor.execute("ALTER TABLE threats ADD COLUMN risk TEXT")
            conn.commit()
        except Exception:
            pass
    conn.close()

def insert_record(record: Dict) -> bool:
    """Insert a normalized threat record into the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
        INSERT INTO threats (
            source, title, summary, threats, sectors, iocs, 
            url, published_at, severity, risk
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            record.get('source', ''),
            record.get('title', ''),
            record.get('summary', ''),
            json.dumps(record.get('threats', [])),
            json.dumps(record.get('sectors', [])),
            json.dumps(record.get('iocs', {})),
            record.get('url'),
            record.get('published_at'),
            record.get('severity', 'Low'),
            json.dumps(record.get('risk', {}))
        ))
        
        conn.commit()
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to insert record: {e}")
        return False
    finally:
        conn.close()

def query_recent(
    limit: int = 100,
    sector: Optional[str] = None,
    severity: Optional[str] = None,
    min_risk: Optional[int] = None,
) -> List[Dict]:
    """Query threats with optional filters"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    query = "SELECT * FROM threats WHERE 1=1"
    params = []
    
    if severity:
        query += " AND severity = ?"
        params.append(severity)
    
    if sector:
        query += " AND sectors LIKE ?"
        params.append(f'%"{sector}"%')

    if min_risk is not None:
        # naive filter on digital risk >= min_risk
        query += " AND (CAST(json_extract(risk, '$.digital') AS INTEGER) >= ?)"
        params.append(int(min_risk))
    
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
            record['risk'] = json.loads(record.get('risk') or '{}')
        except json.JSONDecodeError:
            record['threats'] = []
            record['sectors'] = []
            record['iocs'] = {}
            record['risk'] = {}
        
        results.append(record)
    
    conn.close()
    return results

def get_stats() -> Dict:
    """Get database statistics, including average digital/physical risk"""
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

    # Average risks (digital and physical)
    try:
        cursor.execute("SELECT AVG(CAST(json_extract(risk, '$.digital') AS REAL)) FROM threats WHERE risk IS NOT NULL")
        stats['avg_digital_risk'] = float(cursor.fetchone()[0] or 0)
    except Exception:
        stats['avg_digital_risk'] = 0.0
    try:
        cursor.execute("SELECT AVG(CAST(json_extract(risk, '$.physical') AS REAL)) FROM threats WHERE risk IS NOT NULL")
        stats['avg_physical_risk'] = float(cursor.fetchone()[0] or 0)
    except Exception:
        stats['avg_physical_risk'] = 0.0
    
    conn.close()
    return stats

def clear_database():
    """Clear all records from the database (for testing)"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM threats")
    conn.commit()
    conn.close()
