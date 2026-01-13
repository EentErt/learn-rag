#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, SemanticSearch, chunk_text, semantic_chunk_text, ChunkedSemanticSearch
import json

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="verify semantic search setup")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify embeddings")

    embed_query_parser = subparsers.add_parser("embedquery", help="Embed a given query text")
    embed_query_parser.add_argument("query", type=str, help="Query text to embed")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed a given text")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    search_parser = subparsers.add_parser("search", help="Perform semantic search")
    search_parser.add_argument("query", type=str, help="Query text to search for")
    search_parser.add_argument("--limit", type=int, nargs="?", default=5, help="Number of results to return")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk a given text")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, nargs="?", default=200, help="Size of each chunk")
    chunk_parser.add_argument("--overlap", type=int, nargs="?", default=0, help="Overlap between chunks")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk text semantically")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, nargs="?", default=4, help="Maximum chunk size")
    semantic_chunk_parser.add_argument("--overlap", type=int, nargs="?", default=0, help="Overlap between chunks")

    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Embed text chunks")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Perform chunked semantic search")
    search_chunked_parser.add_argument("query", type=str, help="Query to search for")
    search_chunked_parser.add_argument("--limit", type=int, nargs="?", default=5, help="Number of results to return")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            sem_search = SemanticSearch()
            with open("data/movies.json") as file:
                movies = json.load(file)["movies"]
            embeddings = sem_search.load_or_create_embeddings(movies)
            results = sem_search.search(args.query, args.limit)
            for i in range(len(results)):
                print(f"{i+1}. {results[i]["title"]} (score: {results[i]["score"]:.4f})")
                print(f"   {results[i]["description"]}")
        case "chunk":
            chunks = chunk_text(args.text, args.chunk_size, args.overlap)
            print(f"Chunking {len(args.text)} characters")
            for i in range(len(chunks)):
                print(f"{i+1}. {chunks[i]}")
        case "semantic_chunk":
            chunks = semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            for i in range(len(chunks)):
                print(f"{i+1}. {chunks[i]}")
        case "embed_chunks":
            with open("data/movies.json", "r") as file:
                movies = json.load(file)["movies"]
            sem_search = ChunkedSemanticSearch()
            embeddings = sem_search.load_or_create_chunk_embeddings(movies)
            print(f"Generated {len(embeddings)} chunked embeddings")
        case "search_chunked":
            with open("data/movies.json", "r") as file:
                movies = json.load(file)["movies"]
            sem_search = ChunkedSemanticSearch()
            sem_search.load_or_create_chunk_embeddings(movies)
            results = sem_search.search_chunks(args.query, args.limit)
            for i in range(len(results)):
                print(f"{i+1}. {results[i]["title"]} (score: {results[i]["score"]:.4f})")
                print(f"   {results[i]["document"]}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()