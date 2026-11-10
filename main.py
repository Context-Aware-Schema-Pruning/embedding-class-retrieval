"""Rank the Enslaved Ontology classes for each competency question by embedding similarity.

- Entry point: `python src/cq_class_retrieval/main.py <stage>`; stages: prepare -> embed -> rank -> evaluate (or `all`).
- Each stage lives in its own folder; shared constants and csv helpers are in common/.
- Only `embed` needs a GPU / torch; the other stages run CPU-only.
"""

import argparse
import os
import sys

from common.config import MODELS, DATA, OUT_CQ_CLASS_RETRIEVAL
from embed import run_embed
from evaluate import run_evaluate
from prepare import run_prepare
from rank import run_rank


def build_parser():
    """Build the CLI: a stage name plus the few options every stage shares."""
    parser = argparse.ArgumentParser(description="CQ -> ontology class retrieval with embedding models.")
    parser.add_argument("command", choices=["prepare", "embed", "rank", "evaluate", "all"])
    parser.add_argument("--classes-path", default=os.path.join(DATA, "class_descriptions.json"))
    parser.add_argument("--cqs-path", default=os.path.join(DATA, "cqs.csv"))
    parser.add_argument("--out-dir", default=OUT_CQ_CLASS_RETRIEVAL)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--no-instruction", action="store_true",
                        help="empty query prompt (ablation); outputs get a __noinstr suffix")
    parser.add_argument("--self-test", action="store_true",
                        help="evaluate: check the metric code on oracle/reversed rankings, no embeddings needed")
    return parser


def run_all(args):
    """Run prepare -> embed -> rank -> evaluate; later stages still run for the models that succeeded."""
    if run_prepare(args) != 0:
        return 1
    # embed/rank return 1 if any one model failed, so don't stop on them: the other models still get evaluated.
    statuses = [stage(args) for stage in (run_embed, run_rank, run_evaluate)]
    return 1 if any(statuses) else 0


def main():
    """Parse the command line and run the chosen stage, exiting with its status code."""
    args = build_parser().parse_args()
    stages = {
        "prepare": run_prepare,
        "embed": run_embed,
        "rank": run_rank,
        "evaluate": run_evaluate,
        "all": run_all,
    }
    sys.exit(stages[args.command](args))


if __name__ == "__main__":
    main()
