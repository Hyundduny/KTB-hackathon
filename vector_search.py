import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from vector_store_db import check_vector_store

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

vector_dim = 384
index = faiss.IndexFlatL2(vector_dim)
stored_codes = []

def add_code_to_index(description, code):
    global stored_codes
    embedding = embedding_model.encode(description).reshape(1, -1)
    index.add(embedding)
    stored_codes.append((description, code))

def search_similar_code(description, top_k=1):
    query_embedding = embedding_model.encode(description).reshape(1, -1)
    _, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        if idx < len(stored_codes):
            results.append(stored_codes[idx])
    return results
