import argparse
from lib.hybrid_search import normalize_scores, HybridSearch
import json
from lib.gemini import spell_check, rewrite, expand_query, llm_score, llm_score_batch
import time
from sentence_transformers import CrossEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize scores")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="list of scores to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help= "perform weighted hybrid search")
    weighted_search_parser.add_argument("query", type=str, help="query to search for")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="alpha parameter for adjusting search weighting")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="number of results to return")

    rrf_search_parser= subparsers.add_parser("rrf-search", help="Perform RRF search")
    rrf_search_parser.add_argument("query", type=str, help="Query to search for")
    rrf_search_parser.add_argument("-k", type=int, nargs="?", default=60, help="RRF k parameter")
    rrf_search_parser.add_argument("--limit", type=int, nargs="?", default=5, help="Number of results to return")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Method for reranking search results")


    args = parser.parse_args()

    match args.command:
        case "normalize":
            if len(args.scores) == 0:
                return
            scores = normalize_scores(args.scores)
            for score in scores:
                print(f"* {score:.4f}")
        case "weighted-search":
            with open("data/movies.json", "r") as f:
                movies = json.load(f)['movies']
            search = HybridSearch(movies)
            results = search.weighted_search(args.query, args.alpha, args.limit)
            i = 1
            for idx in results:
                print(f"{i}. {results[idx]['document']['title']}")
                if results[idx].get('llm_score'):
                    print(f"   Rerank Score: {results[idx]['llm_score']}/10")
                print(f"   Hybrid Score: {results[idx]['hybrid_score']:.3f}")
                print(f"   BM25: {results[idx]['bm25_score']:.3f}, Semantic: {results[idx]['semantic_score']:.3f}")
                print(f"   {results[idx]['document']['description'][:100]}...")
                i += 1
        case "rrf-search":
            limit = args.limit
            if args.rerank_method:
                limit *= 5
            query = args.query
            with open("data/movies.json", "r") as f:
                movies = json.load(f)['movies']
            match args.enhance:
                case "spell":
                    query = spell_check(query).lstrip("Corrected: ")
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")
                case "rewrite":
                    query = rewrite(query).lstrip("Rewritten query: ")
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> {query}'\n")
                case "expand":
                    query = expand_query(query)
                    print(f"Enhanced query ({args.enhance}): '{args.query}' => {query}'\n")
                case _:
                    pass
            

                
            search = HybridSearch(movies)
            results = search.rrf_search(query, args.k, limit)
            if args.rerank_method == "individual":
                print(f"Reranking top {args.limit} results using individual method...")
                print(f"Reciprocal Rank Fusion Results for '{args.query}' (k={args.k}):")
                for idx in results:
                    score = float(llm_score(results[idx]['document'], query).lstrip('Score: '))
                    results[idx]['llm_score'] = score
                    time.sleep(10)
                results = dict(sorted(results.items(), key = lambda x: x[1]['llm_score'], reverse = True))
            elif args.rerank_method == "batch":
                print(f"Reranking top {args.limit} results using batch method...")
                doc_list = str([f"ID: {idx}\n{results[idx]['document']['title']}: {results[idx]['document']['description']}" for idx in results])
                score_list = llm_score_batch(doc_list, query)
                print(score_list)
                for idx in results:
                    results[idx]['llm_rank'] = score_list.index(idx)
                results = dict(sorted(results.items(), key = lambda x: x[1]['llm_rank'], reverse = False))
            elif args.rerank_method == "cross_encoder":
                pairs = [[query, f"{results[idx]['document']['title']} - {results[idx]['document']['description']}"] for idx in results]

                cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
                scores = cross_encoder.predict(pairs)
                for i, idx in enumerate(results.keys()):
                    results[idx]['ce_score'] = scores[i]

            
            i = 1
            for idx in results:
                print(f"{i}. {results[idx]['document']['title']}")
                if results[idx].get('llm_score'):
                    print(f"   Rerank Score: {results[idx]['llm_score']:.3f}/10")
                elif results[idx].get('llm_rank'):
                    print(f"   Rerank Rank: {results[idx]['llm_rank']}")
                elif results[idx].get('ce_score'):
                    print(f"   Cross Encoder Score: {results[idx]['ce_score']:.3f}")
                print(f"   RRF Score: {results[idx]['rrf_score']:.3f}")
                print(f"   BM25 Rank: {results[idx]['bm25_rank']:.3f}, Semantic Rank: {results[idx]['semantic_rank']:.3f}")
                print(f"   {results[idx]['document']['description'][:100]}...")
                i += 1
                if i > args.limit:
                    break
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()