#!/usr/bin/env python3

import argparse
import json
from process_string import process_string
from lib.inverted_index import InvertedIndex
from lib.constants import BM25_K1, BM25_B

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build inverted index")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document and term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency count for")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term to get IDF for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF for a document and term")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to get TF-IDF for")

    bm25idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF for a term")
    bm25idf_parser.add_argument("term", type=str, help="Term to get BM25idf for")

    bm25tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF for a document and term")
    bm25tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25tf_parser.add_argument("term", type=str, help="Term to get BM25 TF for")
    bm25tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_K1, help="BM25 K1 parameter")
    bm25tf_parser.add_argument("b", type=float, nargs="?", default=BM25_B, help="BM25 K1 parameter")

    bm25_search_parser = subparsers.add_parser("bm25search", help="Search movies using BM25")
    bm25_search_parser.add_argument("query", type=str, help="Search query")
    bm25_search_parser.add_argument("limit", type=int, nargs="?", default=5, help="Number of results to return")

    args = parser.parse_args()

    inv_index = InvertedIndex()

    results = []

    match args.command:
        case "search":
            try:
                inv_index.load()
            except Exception as e:
                print(e)
            print("searching for:", args.query)

            query = process_string(args.query)

            for token in query:
                docs = inv_index.get_documents(token)
                if docs is None:
                    continue
                results.extend(docs)


        case "build":
            inv_index.build()
            inv_index.save()

        case "tf":
            try:
                inv_index.load()
            except Exception as e:
                print(e)
            freq = inv_index.get_tf(args.doc_id, args.term)
            print(freq)

        case "idf":
            try:
                inv_index.load()
            except Exception as e:
                print(e)

            idf = inv_index.get_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf}")

        case "tfidf":
            try:
                inv_index.load()
            except Exception as e:
                print(e)
            
            tf_idf = inv_index.get_tfidf(args.doc_id, args.term, args.k1, args.b)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case "bm25idf":
            try:
                inv_index.load()
            except Exception as e:
                print(e)

            bm25idf = inv_index.get_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case "bm25tf":
            try:
                inv_index.load()
            except Exception as e:
                print(e)

            bm25tf = inv_index.get_bm25_tf(args.doc_id, args.term)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

        case "bm25search":
            try:
                inv_index.load()
            except Exception as e:
                print(e)

            bm25_search_results = inv_index.bm25_search(args.query, args.limit)
            for i in range(len(bm25_search_results)):
                result = bm25_search_results[i][0]
                score = bm25_search_results[i][1]
                print(f"{i}: ({result["id"]}) {result["title"]} - Score: {score:.2f}")

        case _:
            parser.print_help()

    

    with open("data/movies.json", "r") as file:
        movie_file = json.load(file)


    

    

    
    '''
    for movie in movie_file["movies"]:
        for word in process_string(movie["title"]):
            if any(subquery in word for subquery in query):
                results.append(movie)
                break
    '''


    results = sorted(results, key=lambda movie: movie["id"])
    
    i = 1
    for result in results:
        print(f"{i}: {result["title"]}")
        i += 1
        if i > 5:
            break





if __name__ == "__main__":
    main()