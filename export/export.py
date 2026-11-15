"""Write one rankings/{run}__{view}.csv as JSON: each CQ with its question and its top N classes with their view text."""

import json
import os

from common.config import VIEWS
from common.files import ensure, read_csv
from rerank.rerank import load_ranking


def parse_ranking_name(path):
    """Split a rankings file name into (run, view), e.g. "bge-m3__label.csv" -> ("bge-m3", "label")."""
    file_name = os.path.basename(path)
    name = file_name.removesuffix(".csv")

    parts = name.split("__")
    # The view is always the last part; a run can itself contain "__" ("qwen3-8b__noinstr").
    view = parts[-1]
    run = "__".join(parts[:-1])

    if run == "" or view not in VIEWS:
        raise ValueError(f"{file_name} is not a rankings file named {{run}}__{{view}}.csv")
    return run, view


def build_export(path, ranking, top_classes, class_texts, questions, gold_by_cq):
    """Build the JSON dict for one rankings file: file info, then each CQ with its top classes and missed gold classes."""
    run, view = parse_ranking_name(path)

    # A reranked run is named {retriever}+{reranker}; a plain ranking has no "+".
    if "+" in run:
        retriever, reranker = run.split("+")
    else:
        retriever, reranker = run, None

    cqs = []
    for cq_id, ranked in ranking.items():
        gold = gold_by_cq[cq_id]

        classes = []
        missed_gold = []
        for rank, (name, score) in enumerate(ranked, start=1):
            if rank <= top_classes:
                classes.append({
                    "name": name,
                    view: class_texts[name],
                    "rank": rank,
                    "score": score,
                    "is_gold": name in gold,
                })
            elif name in gold:
                missed_gold.append({"name": name, "rank": rank})

        cqs.append({
            "id": cq_id,
            "question": questions[cq_id],
            "gold_classes": gold,
            "missed_gold": missed_gold,
            "classes": classes,
        })

    return {
        "file_name": os.path.basename(path),
        "path": os.path.abspath(path),
        "run": run,
        "retriever": retriever,
        "reranker": reranker,
        "view": view,
        "top_classes": top_classes,
        "n_classes": len(class_texts),
        "cq": cqs,
    }


def run_export(args):
    """Write args.file as {out_dir}/json/{file name}.json with the top args.top_classes classes per CQ."""
    path = args.file
    if path is None:
        print("error: give the rankings csv with --file")
        return 1
    if not os.path.isfile(path):
        print(f"error: {path} not found")
        return 1
    if args.top_classes < 1:
        print("error: --top-classes must be at least 1")
        return 1

    try:
        _, view = parse_ranking_name(path)
    except ValueError as error:
        print(f"error: {error}")
        return 1

    class_rows = read_csv(os.path.join(args.out_dir, "classes.csv"))
    cq_rows = read_csv(os.path.join(args.out_dir, "cqs.csv"))

    class_texts = {}
    for row in class_rows:
        class_texts[row["name"]] = row[view]

    questions = {}
    gold_by_cq = {}
    for row in cq_rows:
        questions[row["cq_id"]] = row["question"]
        gold_by_cq[row["cq_id"]] = row["gold_classes"].split("|")

    ranking = load_ranking(path)

    # A ranking made before classes.csv / cqs.csv changed would pair CQs with the wrong texts or gold classes.
    if set(ranking) != set(questions):
        print(f"error: {path} has different CQs than cqs.csv; re-run rank")
        return 1
    for cq_id, ranked in ranking.items():
        names = {name for name, _ in ranked}
        if names != set(class_texts):
            print(f"error: {path} has different classes than classes.csv for {cq_id}; re-run rank")
            return 1

    data = build_export(path, ranking, args.top_classes, class_texts, questions, gold_by_cq)

    json_dir = ensure(os.path.join(args.out_dir, "json"))
    file_name = os.path.basename(path).removesuffix(".csv") + ".json"
    out_path = os.path.join(json_dir, file_name)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)

    print(f"wrote {out_path} ({len(data['cq'])} CQs, top {args.top_classes} classes each)")
    return 0
