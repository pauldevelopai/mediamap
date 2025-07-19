"""
Embed content for Highlander application

This script loads all learning modules and generates embeddings for them using sentence-transformers.
The embeddings are saved to a JSON file for later use in semantic search.
"""
import os
import sys
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Ensure scripts/ is in sys.path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
MODULES_PATH = os.path.join(DATA_DIR, 'modules.json')
EMBEDDINGS_PATH = os.path.join(DATA_DIR, 'module_embeddings.json')

# Load modules from JSON or create a placeholder
if os.path.exists(MODULES_PATH):
    with open(MODULES_PATH, 'r') as f:
        modules = json.load(f)
else:
    modules = [
        {"id": "module1", "title": "AI Basics", "description": "Intro to AI for media orgs."},
        {"id": "module2", "title": "AI Ethics", "description": "Ethics in AI adoption."}
    ]
    with open(MODULES_PATH, 'w') as f:
        json.dump(modules, f, indent=2)

# Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = {}
for module in modules:
    text = f"{module['title']} {module['description']}"
    emb = model.encode(text)
    embeddings[module['id']] = np.array(emb, dtype=np.float32).tolist()

with open(EMBEDDINGS_PATH, 'w') as f:
    json.dump(embeddings, f, indent=2)

print(f"Saved {len(embeddings)} module embeddings to {EMBEDDINGS_PATH}")
