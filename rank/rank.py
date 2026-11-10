"""Load each run's saved embeddings, rank all classes per CQ by cosine score and write one csv per run x view."""

import json
import os

import numpy as np

from common.config import VIEWS
from common.files import read_csv, run_name, write_csv


def load_run_embeddings(run_dir):
    """Read one run folder back as (meta, cq_vectors, {view: class_vectors}), checking rows against meta's row order."""
    with open(os.path.join(run_dir, "meta.json"), encoding="utf-8") as handle:
        meta = json.load(handle)
    cq_vectors = np.load(os.path.join(run_dir, "cqs.npy"))
    class_vectors = {view: np.load(os.path.join(run_dir, f"classes__{view}.npy")) for view in VIEWS}

    if cq_vectors.shape[0] != len(meta["cq_ids"]):
        raise ValueError(f"{run_dir}: cqs.npy has {cq_vectors.shape[0]} rows, meta lists {len(meta['cq_ids'])} CQs")
    for view, vectors in class_vectors.items():
        if vectors.shape != (len(meta["class_names"]), cq_vectors.shape[1]):
            raise ValueError(f"{run_dir}: classes__{view}.npy has shape {vectors.shape}, expected "
                             f"({len(meta['class_names'])}, {cq_vectors.shape[1]})")
    return meta, cq_vectors, class_vectors


def rank_classes(cq_vectors, class_vectors, cq_ids, class_names, gold_by_cq):
    """Rank every class for every CQ by cosine score; return rows of cq_id, rank, class, score, is_gold."""
    # Rows are unit length, so the dot product is the cosine similarity.
    scores = cq_vectors @ class_vectors.T
    rows = []
    for cq_index, cq_id in enumerate(cq_ids):
        # Stable sort on the negated scores: highest first, ties keep class order so reruns rank identically.
        order = np.argsort(-scores[cq_index], kind="stable")
        for rank, class_index in enumerate(order, start=1):
            name = class_names[class_index]
            rows.append({
                "cq_id": cq_id,
                "rank": rank,
                "class": name,
                "score": float(scores[cq_index, class_index]),
                "is_gold": int(name in gold_by_cq[cq_id]),
            })
    return rows


def write_rankings(path, rows):
    """Save one run x view ranking (all classes per CQ) as csv: cq_id, rank, class, score, is_gold."""
    # Ranks are fixed before rounding, so 6 decimals only shortens the file; it cannot change any metric.
    write_csv(path,
              [{**row, "score": round(row["score"], 6)} for row in rows],
              ["cq_id", "rank", "class", "score", "is_gold"])


def run_rank(args):
    """Rank all classes per CQ for every view of each chosen run; missing or stale runs are reported and skipped."""
    class_names = [row["name"] for row in read_csv(os.path.join(args.out_dir, "classes.csv"))]
    cq_rows = read_csv(os.path.join(args.out_dir, "cqs.csv"))
    cq_ids = [row["cq_id"] for row in cq_rows]
    gold_by_cq = {row["cq_id"]: set(row["gold_classes"].split("|")) for row in cq_rows}
    rankings_dir = os.path.join(args.out_dir, "rankings")

    ranked, skipped = [], []
    for model_key in args.models:
        run = run_name(model_key, args.no_instruction)
        run_dir = os.path.join(args.out_dir, "embeddings", run)
        if not os.path.isdir(run_dir):
            print(f"{run}: not embedded yet ({run_dir}), skipping")
            skipped.append(run)
            continue
        try:
            meta, cq_vectors, class_vectors = load_run_embeddings(run_dir)
        except (OSError, ValueError, KeyError) as error:
            print(f"{run}: unreadable embeddings, skipping ({type(error).__name__}: {error})")
            skipped.append(run)
            continue
        # Embeddings made before classes.csv / cqs.csv changed would pair scores with the wrong rows.
        if meta["class_names"] != class_names or meta["cq_ids"] != cq_ids:
            print(f"{run}: embedded from different classes/CQs than the prepared csvs; re-run embed. Skipping")
            skipped.append(run)
            continue

        for view in VIEWS:
            rows = rank_classes(cq_vectors, class_vectors[view], cq_ids, class_names, gold_by_cq)
            write_rankings(os.path.join(rankings_dir, f"{run}__{view}.csv"), rows)
        print(f"{run}: ranked {len(cq_ids)} CQs x {len(class_names)} classes for {len(VIEWS)} views")
        ranked.append(run)

    print(f"\nranked:  {', '.join(ranked) or 'none'} -> {rankings_dir}")
    if skipped:
        print(f"skipped: {', '.join(skipped)}")
        return 1
    return 0
