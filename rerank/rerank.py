"""Rerank every run's top-k classes per CQ with one fixed reranker; write rankings/{run}+{reranker}__{view}.csv."""

import os

from common.config import RERANK_TASK, RERANKERS, VIEWS
from common.files import read_csv, rerank_run_name, run_name
from embed.embed import free_gpu, report_device
from rank.rank import write_rankings

BATCH_SIZE = 8

# Template structure follows the model cards. Only the task wording is ours: the card defaults ask whether the
# document answers the whole query, but one class usually covers only part of a CQ.
QWEN3_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
    'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def load_ranking(path):
    """Read one rankings/{run}__{view}.csv back as {cq_id: [(class, score), ...]} in retriever order, best first."""
    # csv values are strings, so sort ranks as ints or "10" would come before "2".
    rows = sorted(read_csv(path), key=lambda row: int(row["rank"]))
    ranking = {row["cq_id"]: [] for row in rows}
    for row in rows:
        ranking[row["cq_id"]].append((row["class"], float(row["score"])))
    return ranking


def load_reranker(spec):
    """Load one RERANKERS entry in its listed dtype as (tokenizer, model), on the GPU if one is available."""
    # score_pairs only knows the Qwen3 prompt and yes/no scoring; fail before a long download, not after.
    if spec["kind"] != "qwen3":
        raise ValueError(f"unsupported reranker kind {spec['kind']!r} for {spec['id']}; only 'qwen3' is implemented")

    # Imported here so the CPU-only stages run without torch / transformers installed.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Left padding puts every pair's last real token at position -1, where score_pairs reads the yes/no logits.
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(spec["id"], torch_dtype=getattr(torch, spec["dtype"]))
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return tokenizer, model


def score_pairs(reranker, query, documents):
    """Score one query against each document with a Qwen3 reranker; return P(yes) per document, in order."""
    import torch

    tokenizer, model = reranker
    texts = [f"{QWEN3_PREFIX}<Instruct>: {RERANK_TASK}\n<Query>: {query}\n<Document>: {document}{QWEN3_SUFFIX}"
             for document in documents]
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")

    scores = []
    for start in range(0, len(texts), BATCH_SIZE):
        # The templates already hold every special token, so the tokenizer must not add its own.
        batch = tokenizer(texts[start:start + BATCH_SIZE], padding=True, add_special_tokens=False,
                          return_tensors="pt").to(model.device)
        with torch.no_grad():
            # Only the next-token prediction after the prompt is needed; logits_to_keep=1 skips the rest of the vocab x length.
            logits = model(**batch, logits_to_keep=1).logits[:, -1, :].float()
        yes_no = torch.softmax(logits[:, [no_id, yes_id]], dim=1)
        scores += yes_no[:, 1].tolist()
    return scores


def merge_rankings(ranking, rerank_scores, top_k, gold_by_cq):
    """Reorder each CQ's top_k classes by reranker score, keep the rest in retriever order; return write_rankings rows."""
    rows = []
    for cq_id, classes in ranking.items():
        names = [name for name, _ in classes[:top_k]]
        scores = rerank_scores[cq_id]
        # strict=True raises if the score count is off, instead of pairing scores with the wrong classes.
        head = list(zip(names, scores, strict=True))
        # Python's sort is stable (also with reverse=True), so equal reranker scores keep retriever order.
        head.sort(key=lambda pair: pair[1], reverse=True)
        # The tail keeps its cosine scores; evaluate only reads rank, so mixing score scales is harmless.
        for rank, (name, score) in enumerate(head + classes[top_k:], start=1):
            rows.append({
                "cq_id": cq_id,
                "rank": rank,
                "class": name,
                "score": float(score),
                "is_gold": int(name in gold_by_cq[cq_id]),
            })
    return rows


def run_rerank(args):
    """Rerank every chosen run's top-k classes per CQ and view with one reranker; missing or stale rankings are skipped."""
    class_rows = read_csv(os.path.join(args.out_dir, "classes.csv"))
    cq_rows = read_csv(os.path.join(args.out_dir, "cqs.csv"))
    class_texts = {view: {row["name"]: row[view] for row in class_rows} for view in VIEWS}
    questions = {row["cq_id"]: row["question"] for row in cq_rows}
    gold_by_cq = {row["cq_id"]: set(row["gold_classes"].split("|")) for row in cq_rows}
    class_names = set(class_texts[VIEWS[0]])
    rankings_dir = os.path.join(args.out_dir, "rankings")
    spec = RERANKERS[args.reranker]
    print(f"== rerank with {args.reranker} ({spec['id']}, {spec['dtype']}), top {args.top_k}")

    # Check every ranking before loading the reranker, so it is never loaded for nothing.
    rankings, skipped = {}, []
    for model_key in args.models:
        run = run_name(model_key, args.no_instruction)
        for view in VIEWS:
            path = os.path.join(rankings_dir, f"{run}__{view}.csv")
            if not os.path.isfile(path):
                print(f"   {run}__{view}: not ranked yet, skipping")
                skipped.append(f"{run}__{view}")
                continue
            ranking = load_ranking(path)
            # A ranking made before classes.csv / cqs.csv changed would pair CQs with the wrong class texts.
            if set(ranking) != set(questions) or any({name for name, _ in classes} != class_names
                                                     for classes in ranking.values()):
                print(f"   {run}__{view}: ranked from different classes/CQs than the prepared csvs; re-run rank. Skipping")
                skipped.append(f"{run}__{view}")
                continue
            rankings[(run, view)] = ranking
    if not rankings:
        print("nothing to rerank")
        return 1

    # A score depends only on (CQ, class text), so a class several runs shortlist in the same view is scored once.
    cache = {}
    written = []
    reranker = None
    try:
        report_device()
        reranker = load_reranker(spec)
        for (run, view), ranking in rankings.items():
            rerank_scores = {}
            for cq_id, classes in ranking.items():
                names = [name for name, _ in classes[:args.top_k]]
                new = [name for name in names if (view, cq_id, name) not in cache]
                if new:
                    scores = score_pairs(reranker, questions[cq_id], [class_texts[view][name] for name in new])
                    for name, score in zip(new, scores, strict=True):
                        cache[(view, cq_id, name)] = score
                rerank_scores[cq_id] = [cache[(view, cq_id, name)] for name in names]
            rows = merge_rankings(ranking, rerank_scores, args.top_k, gold_by_cq)
            out_name = f"{rerank_run_name(run, args.reranker)}__{view}"
            write_rankings(os.path.join(rankings_dir, f"{out_name}.csv"), rows)
            print(f"   {out_name}: written ({len(cache)} pairs scored so far)")
            written.append(out_name)
    # A reranker that won't load or score (VRAM, gated licence, old transformers) is reported, not raised.
    except Exception as error:
        print(f"   FAILED: {type(error).__name__}: {error}")
        skipped.append(args.reranker)
    finally:
        del reranker
        free_gpu()

    print(f"\nreranked: {len(written)} run x view files, {len(cache)} pairs scored -> {rankings_dir}")
    if skipped:
        print(f"skipped:  {', '.join(skipped)}")
        return 1
    return 0
