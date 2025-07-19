"""
Initialize vector store for Highlander application

This script loads the embeddings from the JSON file and initializes a vector store
for semantic search of learning modules.
"""
import os
import json
import sys
import numpy as np
import pickle
from sentence_transformers import util

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Constants
EMBEDDINGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "module_embeddings.json")
VECTOR_STORE_FILE = os.path.join(EMBEDDINGS_DIR, "embeddings_index.json")
VECTOR_STORE_PICKLE = os.path.join(EMBEDDINGS_DIR, "embeddings_index.pkl")

def ensure_dir_exists(directory):
    """Ensure that a directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def init_vector_store():
    """Initialize the vector store from embeddings"""
    # Check if embeddings file exists
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"Embeddings file not found: {EMBEDDINGS_FILE}")
        print("Please run embed_content.py first.")
        return
    
    print(f"Loading embeddings from {EMBEDDINGS_FILE}...")
    
    # Load embeddings
    with open(EMBEDDINGS_FILE, 'r') as f:
        embedding_data = json.load(f)
    
    # Create vector store
    vector_store = {
        "module_ids": [],
        "module_titles": [],
        "embeddings": []
    }
    
    # Extract data
    for module in embedding_data["modules"]:
        vector_store["module_ids"].append(module["id"])
        vector_store["module_titles"].append(module["title"])
        vector_store["embeddings"].append(module["embedding"])
    
    # Convert embeddings to numpy array for efficient search
    vector_store["embeddings_array"] = np.array(vector_store["embeddings"])
    
    # Ensure data directory exists
    ensure_dir_exists(EMBEDDINGS_DIR)
    
    # Save as JSON (without numpy array)
    json_store = {
        "module_ids": vector_store["module_ids"],
        "module_titles": vector_store["module_titles"],
        "embeddings": vector_store["embeddings"]
    }
    
    with open(VECTOR_STORE_FILE, 'w') as f:
        json.dump(json_store, f)
    
    # Save as pickle (with numpy array for efficiency)
    with open(VECTOR_STORE_PICKLE, 'wb') as f:
        pickle.dump(vector_store, f)
    
    print(f"Vector store initialized with {len(vector_store['module_ids'])} modules")
    print(f"Saved to {VECTOR_STORE_FILE} and {VECTOR_STORE_PICKLE}")
    
    # Test a simple search
    test_search(vector_store)

def test_search(vector_store):
    """Test a simple search using the vector store"""
    try:
        from sentence_transformers import SentenceTransformer
        
        print("\nTesting search functionality...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Sample query
        query = "machine learning for beginners"
        query_embedding = model.encode(query)
        
        # Convert to numpy array
        embeddings = np.array(vector_store["embeddings"])
        
        # Calculate cosine similarity
        cos_scores = util.cos_sim(query_embedding, embeddings)[0]
        
        # Get top 3 results
        top_results = np.argsort(-cos_scores)[:3]
        
        print(f"\nTop 3 results for query: '{query}'")
        for idx in top_results:
            print(f"Module ID: {vector_store['module_ids'][idx]}, "
                  f"Title: {vector_store['module_titles'][idx]}, "
                  f"Score: {cos_scores[idx]:.4f}")
    
    except Exception as e:
        print(f"Error testing search: {e}")

if __name__ == "__main__":
    init_vector_store()
