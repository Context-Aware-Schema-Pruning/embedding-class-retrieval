"""Default paths, the class text views, and the embedding and reranker models used across stages."""

import os

# common/config.py -> common -> project root (where main.py lives)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT_CQ_RETRIEVAL = os.path.join(ROOT, "out", "cq_retrieval")
OUT_CQ_CLASS_RETRIEVAL = os.path.join(ROOT, "out", "cq_class_retrieval")

VIEWS = ["label", "description", "label_desc", "label_desc_hier", "description_filtered", "label_desc_hier_neigh"]

RERANK_TOP_K = 20

TASK = "Given a competency question, retrieve the ontology classes needed to answer it"

# Kept apart from TASK: changing TASK would change the Qwen embeddings. Each class usually answers only part of a CQ.
RERANK_TASK = ("Given a competency question, judge whether the ontology class described in the Document "
               "provides information needed to answer at least part of the question")

# query_prompt / doc_prompt follow each model card; the instruction only goes on the question side.
# Each reranker comes from the same family as its embedding model; "kind" picks how rerank/ scores a pair.
MODELS = {
    "qwen3-8b": {
        "id": "Qwen/Qwen3-Embedding-8B",
        "query_prompt": f"Instruct: {TASK}\nQuery: ",
        "doc_prompt": "",
        "dtype": "bfloat16",
        "reranker": {"id": "Qwen/Qwen3-Reranker-8B", "kind": "qwen3", "dtype": "bfloat16"},
    },
    "qwen3-4b": {
        "id": "Qwen/Qwen3-Embedding-4B",
        "query_prompt": f"Instruct: {TASK}\nQuery: ",
        "doc_prompt": "",
        "dtype": "bfloat16",
        "reranker": {"id": "Qwen/Qwen3-Reranker-4B", "kind": "qwen3", "dtype": "bfloat16"},
    },
    "embeddinggemma": {
        "id": "google/embeddinggemma-300m",
        "query_prompt": "task: search result | query: ",
        "doc_prompt": "title: none | text: ",
        # EmbeddingGemma's activations overflow in float16 and it is small, so keep float32.
        "dtype": "float32",
        # Google publishes no EmbeddingGemma reranker; BAAI's Gemma-based one is the closest family match.
        "reranker": {"id": "BAAI/bge-reranker-v2-gemma", "kind": "bge-gemma", "dtype": "float16"},
    },
    "bge-m3": {
        "id": "BAAI/bge-m3",
        "query_prompt": "",
        "doc_prompt": "",
        "dtype": "float16",
        "reranker": {"id": "BAAI/bge-reranker-v2-m3", "kind": "cross-encoder", "dtype": "float16"},
    },
}
