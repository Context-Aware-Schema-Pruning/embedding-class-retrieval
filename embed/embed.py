"""Encode the prepared CQs and class views per model and save them as embeddings/{run}/*.npy plus meta.json."""

import gc
import json
import os

import numpy as np

from common.config import MODELS, VIEWS
from common.files import ensure, read_csv, run_name

BATCH_SIZE = 16


def load_model(spec):
    """Load one MODELS entry as a SentenceTransformer in its listed dtype, on the GPU if one is available."""
    # Imported here so the CPU-only stages run without torch / sentence-transformers installed.
    import torch
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(spec["id"], model_kwargs={"torch_dtype": getattr(torch, spec["dtype"])})


def embed_texts(model, texts, prompt):
    """Encode `texts` with `prompt` prepended (empty = none); return unit-length float32 rows in input order."""
    vectors = model.encode(
        texts,
        prompt=prompt or None,
        batch_size=BATCH_SIZE,
        normalize_embeddings=False,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    # Normalise after the float32 cast: in bfloat16 (~3 significant digits) the norms drift to 0.998-1.002.
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def save_run_embeddings(run_dir, cq_vectors, class_vectors, meta):
    """Check one run's vectors against meta's row order and unit length, then save them as .npy plus meta.json."""
    expected = {"cqs": (cq_vectors, len(meta["cq_ids"]))}
    expected.update({f"classes__{view}": (vectors, len(meta["class_names"])) for view, vectors in class_vectors.items()})
    for name, (vectors, rows) in expected.items():
        if vectors.shape[0] != rows:
            raise ValueError(f"{name}: {vectors.shape[0]} rows, expected {rows}")
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(f"{name}: vector norms range {norms.min():.4f}-{norms.max():.4f}, expected 1")

    ensure(run_dir)
    for name, (vectors, _) in expected.items():
        np.save(os.path.join(run_dir, f"{name}.npy"), vectors)
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)


def free_gpu():
    """Release memory left by a model the caller has already deleted, so the next model fits on the GPU."""
    import torch

    # Collect first: PyTorch can only return blocks to the GPU once no Python object still references them.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_embed(args):
    """Embed the prepared CQs and every class view with each chosen model; a failing model is reported and skipped."""
    class_rows = read_csv(os.path.join(args.out_dir, "classes.csv"))
    cq_rows = read_csv(os.path.join(args.out_dir, "cqs.csv"))
    questions = [row["question"] for row in cq_rows]

    succeeded, failed = [], []
    for model_key in args.models:
        spec = MODELS[model_key]
        run = run_name(model_key, args.no_instruction)
        query_prompt = "" if args.no_instruction else spec["query_prompt"]
        print(f"\n== {run} ({spec['id']}, {spec['dtype']})")

        model = None
        try:
            model = load_model(spec)
            cq_vectors = embed_texts(model, questions, query_prompt)
            class_vectors = {view: embed_texts(model, [row[view] for row in class_rows], spec["doc_prompt"])
                             for view in VIEWS}
            meta = {
                "run": run,
                "model_key": model_key,
                "model_id": spec["id"],
                "dtype": spec["dtype"],
                "no_instruction": args.no_instruction,
                "query_prompt": query_prompt,
                "doc_prompt": spec["doc_prompt"],
                "dimensions": int(cq_vectors.shape[1]),
                "cq_ids": [row["cq_id"] for row in cq_rows],
                "class_names": [row["name"] for row in class_rows],
            }
            run_dir = os.path.join(args.out_dir, "embeddings", run)
            save_run_embeddings(run_dir, cq_vectors, class_vectors, meta)
            print(f"   saved {len(questions)} CQs x {len(class_rows)} classes x {len(VIEWS)} views to {run_dir}")
            succeeded.append(run)
        # A model that won't load or encode (VRAM, gated licence, old transformers) shouldn't stop the others.
        except Exception as error:
            print(f"   FAILED: {type(error).__name__}: {error}")
            failed.append(run)
        finally:
            del model
            free_gpu()

    print(f"\nembedded: {', '.join(succeeded) or 'none'}")
    if failed:
        print(f"failed:   {', '.join(failed)}")
        return 1
    return 0
