import os
import sqlite3

# Get absolute path
basedir = os.path.abspath(os.path.dirname('backend/app.py'))
db_path = os.path.join(basedir, 'instance', 'media_analysis.db')
print('Database path:', db_path)
print('Path exists:', os.path.exists(db_path))
print('Path is absolute:', os.path.isabs(db_path))

# Test direct access
try:
    conn = sqlite3.connect(db_path)
    cursor = conn.execute('SELECT COUNT(*) FROM users')
    count = cursor.fetchone()[0]
    print('Direct access works, users:', count)
    conn.close()
except Exception as e:
    print('Direct access failed:', e)
