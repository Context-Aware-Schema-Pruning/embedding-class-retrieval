"""Rerank each run's top-k classes per CQ with the model's same-family reranker; write rankings/{run}+rerank__{view}.csv."""

import os

from common.config import MODELS, RERANK_TASK, VIEWS
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
BGE_GEMMA_PROMPT = (
    "Given a query A and a passage B describing an ontology class, determine whether the class provides information "
    "needed to answer at least part of the query by providing a prediction of either 'Yes' or 'No'."
)


def load_ranking(path):
    """Read one rankings/{run}__{view}.csv back as {cq_id: [(class, score), ...]} in retriever order, best first."""
    # csv values are strings, so sort ranks as ints or "10" would come before "2".
    rows = sorted(read_csv(path), key=lambda row: int(row["rank"]))
    ranking = {row["cq_id"]: [] for row in rows}
    for row in rows:
        ranking[row["cq_id"]].append((row["class"], float(row["score"])))
    return ranking


def load_reranker(spec):
    """Load one reranker entry in its listed dtype: a CrossEncoder, or (tokenizer, model) for the LLM-based kinds."""
    # Imported here so the CPU-only stages run without torch / transformers installed.
    import torch

    dtype = getattr(torch, spec["dtype"])
    if spec["kind"] == "cross-encoder":
        from sentence_transformers import CrossEncoder

        return CrossEncoder(spec["id"], model_kwargs={"torch_dtype": dtype})

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Left padding puts every pair's last real token at position -1, where score_pairs reads the yes/no logits.
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(spec["id"], torch_dtype=dtype)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return tokenizer, model


def score_pairs(reranker, spec, query, documents):
    """Score one query against each document; return one float per document, in order, higher = more relevant."""
    if spec["kind"] == "cross-encoder":
        pairs = [(query, document) for document in documents]
        scores = reranker.predict(pairs, batch_size=BATCH_SIZE, show_progress_bar=False)
        return [float(score) for score in scores]

    import torch

    tokenizer, model = reranker
    if spec["kind"] == "qwen3":
        texts = [f"{QWEN3_PREFIX}<Instruct>: {RERANK_TASK}\n<Query>: {query}\n<Document>: {document}{QWEN3_SUFFIX}"
                 for document in documents]
        yes_id = tokenizer.convert_tokens_to_ids("yes")
        no_id = tokenizer.convert_tokens_to_ids("no")
    else:
        texts = [f"{tokenizer.bos_token}A: {query}\nB: {document}\n{BGE_GEMMA_PROMPT}" for document in documents]
        yes_id = tokenizer("Yes", add_special_tokens=False)["input_ids"][0]

    scores = []
    for start in range(0, len(texts), BATCH_SIZE):
        # The templates already hold every special token, so the tokenizer must not add its own.
        batch = tokenizer(texts[start:start + BATCH_SIZE], padding=True, add_special_tokens=False,
                          return_tensors="pt").to(model.device)
        with torch.no_grad():
            # Only the next-token prediction after the prompt is needed; logits_to_keep=1 skips the rest of the vocab x length.
            logits = model(**batch, logits_to_keep=1).logits[:, -1, :].float()
        if spec["kind"] == "qwen3":
            yes_no = torch.softmax(logits[:, [no_id, yes_id]], dim=1)
            scores += yes_no[:, 1].tolist()
        else:
            scores += logits[:, yes_id].tolist()
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
    """Rerank every view's top-k classes per CQ with each chosen model's reranker; failing or stale runs are reported."""
    class_rows = read_csv(os.path.join(args.out_dir, "classes.csv"))
    cq_rows = read_csv(os.path.join(args.out_dir, "cqs.csv"))
    class_texts = {view: {row["name"]: row[view] for row in class_rows} for view in VIEWS}
    questions = {row["cq_id"]: row["question"] for row in cq_rows}
    gold_by_cq = {row["cq_id"]: set(row["gold_classes"].split("|")) for row in cq_rows}
    class_names = set(class_texts[VIEWS[0]])
    rankings_dir = os.path.join(args.out_dir, "rankings")

    succeeded, failed, skipped = [], [], []
    for model_key in args.models:
        run = run_name(model_key, args.no_instruction)
        spec = MODELS[model_key]["reranker"]
        print(f"\n== {run} -> {rerank_run_name(run)} ({spec['id']}, {spec['dtype']}, top {args.top_k})")

        # Check every ranking before loading the reranker, so a model is never loaded for nothing.
        rankings = {}
        for view in VIEWS:
            path = os.path.join(rankings_dir, f"{run}__{view}.csv")
            if not os.path.isfile(path):
                print(f"   {view}: not ranked yet ({path}), skipping")
                skipped.append(f"{run}__{view}")
                continue
            ranking = load_ranking(path)
            # A ranking made before classes.csv / cqs.csv changed would pair CQs with the wrong class texts.
            if set(ranking) != set(questions) or any({name for name, _ in classes} != class_names
                                                     for classes in ranking.values()):
                print(f"   {view}: ranked from different classes/CQs than the prepared csvs; re-run rank. Skipping")
                skipped.append(f"{run}__{view}")
                continue
            rankings[view] = ranking
        if not rankings:
            continue

        reranker = None
        try:
            report_device()
            reranker = load_reranker(spec)
            for view, ranking in rankings.items():
                rerank_scores = {}
                for cq_id, classes in ranking.items():
                    documents = [class_texts[view][name] for name, _ in classes[:args.top_k]]
                    rerank_scores[cq_id] = score_pairs(reranker, spec, questions[cq_id], documents)
                rows = merge_rankings(ranking, rerank_scores, args.top_k, gold_by_cq)
                write_rankings(os.path.join(rankings_dir, f"{rerank_run_name(run)}__{view}.csv"), rows)
                print(f"   {view}: reranked top {args.top_k} of {len(ranking)} CQs")
            succeeded.append(rerank_run_name(run))
        # A reranker that won't load or score (VRAM, gated licence, old transformers) shouldn't stop the others.
        except Exception as error:
            print(f"   FAILED: {type(error).__name__}: {error}")
            failed.append(rerank_run_name(run))
        finally:
            del reranker
            free_gpu()

    print(f"\nreranked: {', '.join(succeeded) or 'none'} -> {rankings_dir}")
    if failed:
        print(f"failed:   {', '.join(failed)}")
    if skipped:
        print(f"skipped:  {', '.join(skipped)}")
    return 1 if failed or skipped else 0
