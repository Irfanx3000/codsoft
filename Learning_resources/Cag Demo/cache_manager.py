# cache_manager.py
import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

CACHE_FILE = "cache.json"

# Create file if it doesn't exist
if not os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "w") as f:
        json.dump([], f)


def load_cache():
    with open(CACHE_FILE, "r") as f:
        return json.load(f)


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4)


def search_cache(query_embedding, threshold=0.75):
    """
    Returns cached answer if semantically similar question exists.
    """
    cache = load_cache()
    best_score = 0
    best_answer = None

    query_emb = np.array(query_embedding).reshape(1, -1)

    for item in cache:
        emb = np.array(item["embedding"]).reshape(1, -1)
        score = cosine_similarity(query_emb, emb)[0][0]

        if score > best_score:
            best_score = score
            best_answer = item["response"]

    if best_score >= threshold:
        print(f"CACHE HIT (score: {best_score:.2f})")
        return best_answer

    return None


def add_to_cache(query, embedding, response):
    cache = load_cache()
    cache.append({
        "query": query,
        "embedding": embedding,
        "response": response
    })
    save_cache(cache)
