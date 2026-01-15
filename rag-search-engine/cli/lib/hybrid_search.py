import os
from lib.inverted_index import InvertedIndex
from lib.semantic_search import ChunkedSemanticSearch

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_scores = self._bm25_search(query, limit * 500)
        semantic_scores = self.semantic_search.search_chunks(query, limit * 500)

        bm25_norms = normalize_scores([score for _, score in bm25_scores])
        bm25_scores = [(bm25_scores[i][0], bm25_norms[i]) for i in range(len(bm25_scores))]

        sem_norms = normalize_scores([score['score'] for score in semantic_scores])
        for i in range(len(semantic_scores)):
            semantic_scores[i]['score'] = sem_norms[i]


        combined_scores = {}

        for bm25 in bm25_scores:
            if bm25[0]['id'] not in combined_scores:
                combined_scores[bm25[0]['id']] = {
                    "document": bm25[0],
                    "bm25_score": bm25[1],
                    "semantic_score": 0.0,
                    "hybrid_score": 0.0
                }
        for sem in semantic_scores:
            if sem['id'] not in combined_scores:
                combined_scores[sem['id']] = {
                    "document": sem['document'],
                    "bm25_score": 0.0,
                    "semantic_score": sem['score'],
                    "hybrid_score": 0.0
                }
            else:
                combined_scores[sem['id']]['semantic_score'] = sem['score']

        for idx in combined_scores:
            combined_scores[idx]['hybrid_score'] = hybrid_score(combined_scores[idx]['bm25_score'], combined_scores[idx]['semantic_score'], alpha)

        sorted_combined = dict(sorted(combined_scores.items(), key = lambda x: x[1]['hybrid_score'], reverse=True)[:limit])
        return sorted_combined

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search not implemented yet")

def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def normalize_scores(scores):
    if max(scores) == min(scores):
        return [1.0 for _ in scores]
    else:
        norm_scores = []
        for score in scores:
            norm_scores.append((score - min(scores)) / (max(scores) - min(scores)))
        return norm_scores
