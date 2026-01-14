from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
import re

class SemanticSearch():
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query = self.generate_embedding(query)
        similarities = []
        for i in range(len(self.embeddings)):
            similarity = cosine_similarity(query, self.embeddings[i])
            similarities.append((similarity, self.documents[i]))

        similarities.sort(key=lambda x: x[0], reverse=True)
        result = []
        for i in range(limit):
            result.append({
                "score": similarities[i][0],
                "title": similarities[i][1]["title"],
                "description": similarities[i][1]["description"]
            })
        return result

    def generate_embedding(self, text):
        if text.strip() == "":
            raise ValueError("Input text cannot be empty.")

        embedding = self.model.encode([text])

        return embedding[0]

    def load_or_create_embeddings(self, documents):
        try:
            with open("cache/movie_embeddings.npy", "rb") as file:
                self.embeddings = np.load(file)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        except:
            return self.build_embeddings(documents)

    def build_embeddings(self, documents):
        self.documents = documents
        doc_list = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            doc_list.append( f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(doc_list, show_progress_bar=True)
        if not os.path.exists("cache"):
            os.makedirs("cache")
        with open("cache/movie_embeddings.npy", "wb") as file:
            np.save(file, self.embeddings)
        return self.embeddings

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        doc_list = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            doc_list.append( f"{doc['title']}: {doc['description']}")
        chunks = []
        chunks_metadata = []
        for idx, doc in enumerate(documents):
            if doc.get("description") == "":
                continue
            doc_chunks = semantic_chunk_text(doc["description"], max=4, overlap=1)
            chunks.extend(doc_chunks)
            for chunk_idx, chunk in enumerate(doc_chunks):
                chunks_metadata.append({
                    "movie_idx": idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(doc_chunks)
                })
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = chunks_metadata
        with open("cache/chunk_embeddings.npy", "wb") as embeddings_file:
            np.save(embeddings_file, self.chunk_embeddings)
        with open("cache/chunk_metadata.json", "w") as metadata_file:
            json.dump({"chunks": chunks_metadata, "total_chunks": len(chunks)}, metadata_file, indent=2)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents):
        try:
            with open("cache/chunk_embeddings.npy", "rb") as file:
                self.chunk_embeddings = np.load(file)
            with open("cache/chunk_metadata.json", "r") as file:
                self.chunk_metadata = json.load(file)["chunks"]
            self.documents = documents
            return self.chunk_embeddings
        except:
            return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        query = self.generate_embedding(query)
        chunk_scores = []
        for i, embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query, embedding)
            chunk_scores.append({
                "chunk_idx": i,
                "movie_idx": self.chunk_metadata[i]["movie_idx"],
                "score": score
            })
        movie_scores = {}
        for score in chunk_scores:
            if not movie_scores.get(score["movie_idx"]):
                movie_scores[score["movie_idx"]] = score["score"]
            elif score["score"] > movie_scores[score["movie_idx"]]:
                movie_scores[score["movie_idx"]] = score["score"]
        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        result = []
        for (movie_id, score) in sorted_movies[:limit]:
            result.append({
                "id": movie_id,
                "title": self.documents[movie_id]["title"],
                "document": self.documents[movie_id]["description"][:100],
                "score": round(score, 4),
                "metadata": {}
            })
        return result
        




def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def embed_query_text(query):
    sem_search = SemanticSearch()
    embedding = sem_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def verify_embeddings():
    sem_search = SemanticSearch()
    with open("data/movies.json") as file:
        movies = json.load(file)["movies"]
    embeddings = sem_search.load_or_create_embeddings(movies)
    print(f"Number of docs: {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_text(text):
    sem_search = SemanticSearch()
    embedding = sem_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def chunk_text(text, chunk_size, overlap):
    words = text.split()
    chunks = []
    over = ""
    while len(words) > chunk_size:
        chunk = ""
        if over:
            chunk = over + " "
        chunk += " ".join(words[:chunk_size])
        chunks.append(chunk)
        if overlap > 0:
            over = " ".join(words[chunk_size - overlap:chunk_size])
        words = words[chunk_size:]
    if over:
        chunks.append(over + " " +" ".join(words))
    else:
        chunks.append(" ".join(words))
    return chunks

def semantic_chunk_text(text, max, overlap):
    text = text.strip()
    if text == "":
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 1 and not sentences[0].endswith((".", "!", "?")):
        return [text]
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = ""
        chunk = sentences[i:i+max]
        if len(chunk) <= overlap:
            break
        i += max
        if overlap:
            i -= overlap
        for sentence in chunk:
            sentence = sentence.strip()
        if chunk == "":
            continue
        chunks.append(" ".join(chunk))
    return chunks


