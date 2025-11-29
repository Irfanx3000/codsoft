# cag.py
import json
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from embeddings import get_embedding
from llm import generate_answer

THRESHOLD = 0.75
DATASET_FILE = "dataset.json"


# -------------------------------
# Load + Save Dataset
# -------------------------------
def load_dataset():
    with open(DATASET_FILE, "r") as f:
        return json.load(f)

def save_dataset(data):
    with open(DATASET_FILE, "w") as f:
        json.dump(data, f, indent=4)


# -------------------------------
# Embed Dataset
# -------------------------------
def embed_dataset():
    dataset = load_dataset()
    db = []
    for item in dataset:
        emb = get_embedding(item["question"])
        db.append({
            "question": item["question"],
            "answer": item["answer"],
            "embedding": emb
        })
    return db

vector_db = embed_dataset()   # in-memory vector DB


# -------------------------------
# Search Function (with timing)
# -------------------------------
def search_dataset(query):
    print("\n--- Searching Dataset ---")

    start = time.time()
    q_emb = np.array(get_embedding(query)).reshape(1, -1)

    best_score = 0.0
    best_item = None

    for item in vector_db:
        emb = np.array(item["embedding"]).reshape(1, -1)
        score = float(cosine_similarity(q_emb, emb)[0][0])

        if score > best_score:
            best_score = score
            best_item = item

    elapsed = (time.time() - start) * 1000  # ms

    print(f"Search Time: {elapsed:.2f} ms")
    print(f"Best Similarity: {best_score:.4f}")

    if best_score >= THRESHOLD:
        print("CACHE HIT!")
        return best_item["answer"]

    print("CACHE MISS!")
    return None


# -------------------------------
# Add New Cache Entry
# -------------------------------
def add_to_dataset(query, answer):
    print("Caching new answer locally...")

    data = load_dataset()
    data.append({"question": query, "answer": answer})
    save_dataset(data)

    vector_db.append({
        "question": query,
        "answer": answer,
        "embedding": get_embedding(query)
    })


# -------------------------------
# CAG Pipeline
# -------------------------------
def cag_pipeline(query):
    print("\n==============================")
    print("User Query:", query)
    print("==============================")

    # 1. Try Cache
    cached_answer = search_dataset(query)
    if cached_answer:
        return cached_answer

    # 2. Fallback (dummy LLM)
    print("\n--- Calling Offline LLM ---")
    start = time.time()
    llm_response = generate_answer(query)
    elapsed = (time.time() - start) * 1000
    print(f"Offline Generation Time: {elapsed:.2f} ms")

    # 3. Store into dataset (cache)
    add_to_dataset(query, llm_response)

    return llm_response
