"""Score each run x view ranking per CQ (recall/precision@k, R-precision, MRR) against gold and a random baseline."""

import math
import os
import random
import tempfile

from common.config import VIEWS
from common.files import read_csv, run_name, write_csv
from rank import write_rankings

RECALL_KS = [5, 10, 20]
PRECISION_KS = [5, 10]
METRICS = [f"recall@{k}" for k in RECALL_KS] + [f"precision@{k}" for k in PRECISION_KS] + ["r_precision", "mrr"]


def metrics_for_cq(ranked_classes, gold):
    """Score one CQ's ranking (class names, best first) against its gold set: recall/precision@k, R-precision, MRR."""
    hits = [name in gold for name in ranked_classes]
    metrics = {}
    for k in RECALL_KS:
        metrics[f"recall@{k}"] = sum(hits[:k]) / len(gold)
    for k in PRECISION_KS:
        metrics[f"precision@{k}"] = sum(hits[:k]) / k
    # R-precision cuts at the CQ's own gold size, so a perfect ranking scores 1.0 whatever that size is.
    metrics["r_precision"] = sum(hits[:len(gold)]) / len(gold)
    first_hit = next((rank for rank, hit in enumerate(hits, start=1) if hit), None)
    metrics["mrr"] = 1 / first_hit if first_hit else 0.0
    # Listed in rank order (nearest miss first) because gold is a set and has no stable order of its own.
    metrics["missed_top10"] = [name for name in ranked_classes[10:] if name in gold]
    return metrics


def random_baseline(gold_sizes, n_classes):
    """Expected metrics of a uniformly random ranking, per CQ then averaged, so it lines up with the real runs."""
    per_cq = []
    for g in gold_sizes:
        metrics = {f"recall@{k}": k / n_classes for k in RECALL_KS}
        metrics.update({f"precision@{k}": g / n_classes for k in PRECISION_KS})
        metrics["r_precision"] = g / n_classes
        # P(first gold at rank r) = C(n - r, g - 1) / C(n, g): the other g - 1 gold classes all fall after rank r.
        metrics["mrr"] = sum(math.comb(n_classes - r, g - 1) / r for r in range(1, n_classes - g + 2)) / math.comb(n_classes, g)
        per_cq.append(metrics)
    return {name: sum(m[name] for m in per_cq) / len(per_cq) for name in per_cq[0]}


def summarise(rankings_dir, runs, gold_by_cq, n_classes):
    """Score every run x view ranking file; return (summary rows incl. a random baseline, per-CQ rows, missing files)."""
    summary_rows, per_cq_rows, missing = [], [], []
    for run in runs:
        for view in VIEWS:
            path = os.path.join(rankings_dir, f"{run}__{view}.csv")
            if not os.path.isfile(path):
                missing.append(f"{run}__{view}")
                continue
            ranked_by_cq = {}
            # csv values are strings, so sort ranks as ints or "10" would come before "2".
            for row in sorted(read_csv(path), key=lambda row: int(row["rank"])):
                ranked_by_cq.setdefault(row["cq_id"], []).append(row["class"])
            if set(ranked_by_cq) != set(gold_by_cq):
                raise ValueError(f"{path}: CQs differ from the prepared cqs.csv; re-run rank")

            cq_metrics = []
            for cq_id, gold in gold_by_cq.items():
                metrics = metrics_for_cq(ranked_by_cq[cq_id], gold)
                cq_metrics.append(metrics)
                per_cq_rows.append({"run": run, "view": view, "cq_id": cq_id,
                                    **{name: round(metrics[name], 4) for name in METRICS},
                                    "missed_top10": "|".join(metrics["missed_top10"])})
            summary_rows.append({"run": run, "view": view,
                                 **{name: round(sum(m[name] for m in cq_metrics) / len(cq_metrics), 4)
                                    for name in METRICS}})

    baseline = random_baseline([len(gold) for gold in gold_by_cq.values()], n_classes)
    summary_rows.append({"run": "random", "view": "-", **{name: round(baseline[name], 4) for name in METRICS}})
    return summary_rows, per_cq_rows, missing


def print_summary(summary_rows, n_cqs):
    """Print the run x view table sorted by Recall@10 (random baseline last), with the caveats for reading it."""
    headers = {"recall@5": "R@5", "recall@10": "R@10", "recall@20": "R@20", "precision@5": "P@5",
               "precision@10": "P@10", "r_precision": "R-Prec", "mrr": "MRR"}
    runs = sorted((row for row in summary_rows if row["run"] != "random"), key=lambda row: -row["recall@10"])
    baseline = [row for row in summary_rows if row["run"] == "random"]

    run_width = max(len(row["run"]) for row in summary_rows)
    view_width = max(len(row["view"]) for row in summary_rows + [{"view": "view"}])
    print(f"\n{'run':<{run_width}}  {'view':<{view_width}}  " + "  ".join(f"{headers[m]:>6}" for m in METRICS))
    for row in runs + baseline:
        print(f"{row['run']:<{run_width}}  {row['view']:<{view_width}}  "
              + "  ".join(f"{row[m]:>6.3f}" for m in METRICS))

    print(f"\nNotes: averaged over {n_cqs} CQs, so differences below about 0.05 are noise.")
    print("       Class descriptions were written using the CQ wording, so these scores are optimistic.")
    print("       R-Prec cuts at each CQ's own gold size; random = expected score of a random ranking.")


def self_test():
    """Check the metric code on perfect and reversed rankings with known answers, incl. a csv round trip; 0 = pass."""
    n = 44
    names = [f"C{i:02d}" for i in range(n)]
    failures = []

    def check(label, got, want):
        if abs(got - want) > 1e-4:
            failures.append(f"{label}: got {got:.4f}, expected {want:.4f}")

    gold_sizes = [1, 6, 10, 13, 44]
    for g in gold_sizes:
        gold = set(names[:g])
        perfect = metrics_for_cq(names, gold)
        reversed_ = metrics_for_cq(names[::-1], gold)
        # Reversed: the gold classes fill the last g ranks, so the top k holds max(0, k - (n - g)) of them.
        for k in RECALL_KS:
            check(f"perfect g={g} recall@{k}", perfect[f"recall@{k}"], min(k, g) / g)
            check(f"reversed g={g} recall@{k}", reversed_[f"recall@{k}"], max(0, k - (n - g)) / g)
        for k in PRECISION_KS:
            check(f"perfect g={g} precision@{k}", perfect[f"precision@{k}"], min(k, g) / k)
            check(f"reversed g={g} precision@{k}", reversed_[f"precision@{k}"], max(0, k - (n - g)) / k)
        check(f"perfect g={g} r_precision", perfect["r_precision"], 1.0)
        check(f"reversed g={g} r_precision", reversed_["r_precision"], max(0, g - (n - g)) / g)
        check(f"perfect g={g} mrr", perfect["mrr"], 1.0)
        check(f"reversed g={g} mrr", reversed_["mrr"], 1 / (n - g + 1))

    # Round trip through write_rankings -> summarise with rows shuffled, so ranks must be re-sorted as numbers.
    gold_by_cq = {f"T{g}": set(names[:g]) for g in gold_sizes}
    rows = [{"cq_id": cq_id, "rank": rank, "class": name, "score": 1 - rank / n, "is_gold": int(name in gold)}
            for cq_id, gold in gold_by_cq.items() for rank, name in enumerate(names, start=1)]
    random.Random(0).shuffle(rows)
    with tempfile.TemporaryDirectory() as tmp:
        write_rankings(os.path.join(tmp, f"perfect__{VIEWS[0]}.csv"), rows)
        _, per_cq_rows, _ = summarise(tmp, ["perfect"], gold_by_cq, n)
    for row in per_cq_rows:
        check(f"round trip {row['cq_id']} r_precision", row["r_precision"], 1.0)
        check(f"round trip {row['cq_id']} mrr", row["mrr"], 1.0)

    if failures:
        print(f"self-test FAILED ({len(failures)}):")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(f"self-test passed: perfect and reversed rankings for gold sizes {gold_sizes}, plus a csv round trip")
    return 0


def run_evaluate(args):
    """Score the chosen runs' rankings, write results/summary.csv and per_cq.csv, and print the summary table."""
    if args.self_test:
        return self_test()

    n_classes = len(read_csv(os.path.join(args.out_dir, "classes.csv")))
    gold_by_cq = {row["cq_id"]: set(row["gold_classes"].split("|"))
                  for row in read_csv(os.path.join(args.out_dir, "cqs.csv"))}
    rankings_dir = os.path.join(args.out_dir, "rankings")

    runs = []
    for model_key in args.models:
        runs.append(run_name(model_key, args.no_instruction))
        # summary.csv is rewritten each time, so pull in an existing __noinstr run to keep the ablation in the same table.
        ablation = run_name(model_key, True)
        if not args.no_instruction and os.path.isfile(os.path.join(rankings_dir, f"{ablation}__{VIEWS[0]}.csv")):
            runs.append(ablation)

    try:
        summary_rows, per_cq_rows, missing = summarise(rankings_dir, runs, gold_by_cq, n_classes)
    except ValueError as error:
        print(f"error: {error}")
        return 1
    if not per_cq_rows:
        print(f"no rankings found in {rankings_dir} for: {', '.join(runs)}; run rank first")
        return 1

    results_dir = os.path.join(args.out_dir, "results")
    write_csv(os.path.join(results_dir, "summary.csv"), summary_rows, ["run", "view"] + METRICS)
    write_csv(os.path.join(results_dir, "per_cq.csv"), per_cq_rows, ["run", "view", "cq_id"] + METRICS + ["missed_top10"])
    print_summary(summary_rows, len(gold_by_cq))
    print(f"\nwrote {os.path.join(results_dir, 'summary.csv')}\n      {os.path.join(results_dir, 'per_cq.csv')}")

    if missing:
        print(f"missing rankings (not evaluated): {', '.join(missing)}")
        return 1
    return 0
