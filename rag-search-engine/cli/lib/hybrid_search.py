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
            print("path does not exist")
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

        sem_norms = normalize_scores([score for _, score in semantic_scores])
        semantic_scores = [(semantic_scores[i][0], sem_norms[i]) for i in range(len(semantic_scores))]

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
            if sem[0]['id'] not in combined_scores:
                combined_scores[sem[0]['id']] = {
                    "document": sem[0],
                    "bm25_score": 0.0,
                    "semantic_score": sem[1],
                    "hybrid_score": 0.0
                }
            else:
                combined_scores[sem[0]['id']]['semantic_score'] = sem[1]

        for idx in combined_scores:
            combined_scores[idx]['hybrid_score'] = hybrid_score(combined_scores[idx]['bm25_score'], combined_scores[idx]['semantic_score'], alpha)

        sorted_combined = dict(sorted(combined_scores.items(), key = lambda x: x[1]['hybrid_score'], reverse=True)[:limit])
        return sorted_combined

    def rrf_search(self, query, k, limit=10):
        bm25_scores = self._bm25_search(query, limit * 500)
        semantic_scores = self.semantic_search.search_chunks(query, limit * 500)

        combined_scores = {}
        for i, bm25 in enumerate(bm25_scores):
            if bm25[0]['id'] not in combined_scores:
                combined_scores[bm25[0]['id']] = {
                    "document": bm25[0],
                    "bm25_rank": 1 / (k + i + 1),
                    "semantic_rank": 0.0,
                    "rrf_score": 0.0
                }

        for i, sem in enumerate(semantic_scores):
            if sem[0]['id'] not in combined_scores:
                combined_scores[sem[0]['id']] = {
                    "document": sem[0],
                    "bm25_rank": 0.0,
                    "semantic_rank": 1 / (k + i + 1),
                    "rrf_score": 0.0
                }
            else:
                combined_scores[sem[0]['id']]['semantic_rank'] = 1 / (k + i + 1)

        for idx in combined_scores:
            combined_scores[idx]['rrf_score'] = combined_scores[idx]['bm25_rank'] + combined_scores[idx]['semantic_rank']

        sorted_combined = dict(sorted(combined_scores.items(), key = lambda x: x[1]['rrf_score'], reverse=True)[:limit])
        return sorted_combined


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
