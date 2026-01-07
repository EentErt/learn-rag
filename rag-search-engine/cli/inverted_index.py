from process_string import process_string
import json
import pickle
import os
from collections import Counter
import math
from constants import BM25_K1

class InvertedIndex():
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}

    def __add_document(self, doc_id, text):
        tokens = process_string(text)
        self.term_frequencies[doc_id] = Counter()
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
        for id in doc_ids:
            docs.append(self.docmap[id])
        return docs
    
    def get_tf(self, doc_id, term):
        term = process_string(term)
        if len(term) != 1:
            raise Exception("search term must be a single word")
        
        return self.term_frequencies.get(doc_id)[term[0]]
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1):
        raw_tf = self.get_tf(doc_id, term)
        bm25_tf = (raw_tf * (k1 + 1)) / (raw_tf + k1)
        return bm25_tf


    
    def get_idf(self, term):
        term = process_string(term)
        if len(term) != 1:
            raise Exception("search term must be a single word")
        
        doc_count = len(self.docmap)
        match_doc_count = len(self.get_documents(term[0]))
        idf = math.log((doc_count + 1) / (match_doc_count + 1))

        return idf

    def get_bm25_idf(self, term: str) -> float:
        term = process_string(term)
        if len(term) != 1:
            raise Exception("search term must be a single word")
        
        n = len(self.docmap)
        df = len(self.get_documents(term[0]))
        return math.log((n - df + 0.5) / (df + 0.5) + 1)
    
    def get_tfidf(self, doc_id, term):
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)

        return tf * idf
    
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

    def load(self):
        if not os.path.exists("cache"):
            raise Exception("cache folder does not exist")
        elif not os.path.exists("cache/docmap.pkl"):
            raise Exception("docmap file does not exist")
        elif not os.path.exists("cache/index.pkl"):
            raise Exception("index file does not exist")
        elif not os.path.exists("cache/term_frequencies.pkl"):
            raise Exception("term frequencies file does not exist")
        with open("cache/docmap.pkl", "rb") as docmap_file:
            self.docmap = pickle.load(docmap_file)
        with open("cache/index.pkl", "rb") as index_file:
            self.index = pickle.load(index_file)
        with open("cache/term_frequencies.pkl", "rb") as freq_file:
            self.term_frequencies = pickle.load(freq_file)
        


        

    