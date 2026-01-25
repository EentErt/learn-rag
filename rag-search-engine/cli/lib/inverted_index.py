from process_string import process_string
import json
import pickle
import os
from collections import Counter
import math
from lib.constants import BM25_K1, BM25_B

class InvertedIndex():
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}
        self.index_path = "cache/index.pkl"

    def __add_document(self, doc_id, text):
        tokens = process_string(text)
        self.term_frequencies[doc_id] = Counter()
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            self.term_frequencies[doc_id][token] += 1
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term):
        docs = []
        doc_ids = self.index.get(term.lower())
        if doc_ids is None:
            return None
        for idx in doc_ids:
            docs.append(self.docmap[idx])
        return docs
    
    def bm25_search(self, query, limit):
        query = process_string(query)
        scores = {}
        for term in query:
            if self.index.get(term) is None:
                continue
            for doc_id in self.index[term]:
                if scores.get(doc_id) is None:
                    scores[doc_id] = 0.0
                scores[doc_id] += self.bm25(doc_id, term)
        sorted_scores = sorted(scores.items(), key = lambda x: x[1], reverse = True)
        result = []
        for doc in sorted_scores[:limit]:
            result.append((self.docmap[doc[0]], doc[1]))
        return result

    
    def bm25(self, doc_id, term):
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)
        return tf * idf
    
    def get_tf(self, doc_id, term):
        term = process_string(term)
        if len(term) != 1:
            raise Exception("search term must be a single word")
        
        return self.term_frequencies.get(doc_id)[term[0]]
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        raw_tf = self.get_tf(doc_id, term)
        bm25_tf = (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm)
        return bm25_tf
    
    def get_idf(self, term):
        term = process_string(term)
        if len(term) != 1:
            raise Exception("search term must be a single word")
        match_doc_count = 0
        doc_count = len(self.docmap)
        if self.get_documents(term[0]):
            match_doc_count = len(self.get_documents(term[0]))
        idf = math.log((doc_count + 1) / (match_doc_count + 1))

        return idf

    def get_bm25_idf(self, term: str) -> float:
        term = process_string(term)
        if len(term) != 1:
            raise Exception("search term must be a single word")
        
        n = len(self.docmap)
        df = 0
        if self.get_documents(term[0]):
            df = len(self.get_documents(term[0]))
        return math.log((n - df + 0.5) / (df + 0.5) + 1)
    
    def get_tfidf(self, doc_id, term):
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)

        return tf * idf
    
    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        total_length = sum(self.doc_lengths.values())
        return total_length / len(self.doc_lengths)
    
    def build(self):
        with open("data/movies.json") as file:
            movies_obj = json.load(file)
        movies = movies_obj["movies"]

        for movie in movies:
            self.__add_document(movie["id"], f"{movie["title"]} {movie["description"]}")
            self.docmap[movie["id"]] = movie

    def save(self):
        if not os.path.exists("cache"):
            os.makedirs("cache")
        with open("cache/docmap.pkl", "wb") as docmap_file:
            pickle.dump(self.docmap, docmap_file)
        with open("cache/index.pkl", "wb") as index_file:
            pickle.dump(self.index, index_file)
        with open("cache/term_frequencies.pkl", "wb") as freq_file:
            pickle.dump(self.term_frequencies, freq_file)
        with open("cache/doc_lengths.pkl", "wb") as doc_lengths_file:
            pickle.dump(self.doc_lengths, doc_lengths_file)

    def load(self):
        if not os.path.exists("cache"):
            raise Exception("cache folder does not exist")
        elif not os.path.exists("cache/docmap.pkl"):
            raise Exception("docmap file does not exist")
        elif not os.path.exists("cache/index.pkl"):
            raise Exception("index file does not exist")
        elif not os.path.exists("cache/term_frequencies.pkl"):
            raise Exception("term frequencies file does not exist")
        elif not os.path.exists("cache/doc_lengths.pkl"):
            raise Exception("doc lengths file does not exist")
        with open("cache/docmap.pkl", "rb") as docmap_file:
            self.docmap = pickle.load(docmap_file)
        with open("cache/index.pkl", "rb") as index_file:
            self.index = pickle.load(index_file)
        with open("cache/term_frequencies.pkl", "rb") as freq_file:
            self.term_frequencies = pickle.load(freq_file)
        with open("cache/doc_lengths.pkl", "rb") as doc_lengths_file:
            self.doc_lengths = pickle.load(doc_lengths_file)
        


        

    