import argparse
import json
from lib.hybrid_search import HybridSearch

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument("--limit", type=int, default=5, help="Number of results to evaluate (k for precision@k, recall@k")

    args = parser.parse_args()
    limit = args.limit

    with open("data/golden_dataset.json", "r") as f:
        gold = json.load(f)
    

    with open("data/movies.json", "r") as f:
        movies = json.load(f)['movies']

    precision_set = []
    recall_set = []
    f1_set = []
    results = {}
    search = HybridSearch(movies)
    for case in gold['test_cases']:
        results = search.rrf_search(case['query'], 60, limit)
        relevant = 0
        for idx in results:
            if results[idx]['document']['title'] in case['relevant_docs']:
                relevant += 1
        precision = relevant / len(results)
        precision_set.append(precision)
        recall = relevant / len(case['relevant_docs'])
        recall_set.append(recall)
        f1_set.append( 2 * precision * recall / (precision + recall))

    print(f"k={limit}\n")
    for i, case in enumerate(gold['test_cases']):
        print(f"- Query: {case['query']}")
        print(f"  - Precision@{limit}: {precision_set[i]:.4f}")
        print(f"  - Recall@{limit}: {recall_set[i]:.4f}")
        print(f"  - F1 Score: {f1_set[i]:.4f}")
        print(f"  - Retrieved: {", ".join([results[idx]['document']['title'] for idx in results])}")
        print(f"  - Relevant: {", ".join(case['relevant_docs'])}")
        print()



if __name__ == "__main__":
    main()