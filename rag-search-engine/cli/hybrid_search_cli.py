import argparse
from lib.hybrid_search import normalize_scores, HybridSearch
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize scores")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="list of scores to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help= "prerform weighted hybrid search")
    weighted_search_parser.add_argument("query", type=str, help="query to search for")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="alpha parameter for adjusting search weighting")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="number of results to return")

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
                print(f"   Hybrid Score: {results[idx]['hybrid_score']:.3f}")
                print(f"   BM25: {results[idx]['bm25_score']:.3f}, Semantic: {results[idx]['semantic_score']:.3f}")
                print(f"   {results[idx]['document']['description'][:100]}...")
                i += 1
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()