"""Default paths, the class text views and the embedding models used across stages."""

import os

# cq_class_retrieval/common/config.py -> common -> cq_class_retrieval (project root, where main.py lives)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT_CQ_RETRIEVAL = os.path.join(ROOT, "out", "cq_retrieval")
OUT_CQ_CLASS_RETRIEVAL = os.path.join(ROOT, "out", "cq_class_retrieval")

VIEWS = ["label", "description", "label_desc", "label_desc_hier", "description_filtered", "label_desc_hier_neigh"]

TASK = "Given a competency question, retrieve the ontology classes needed to answer it"

# query_prompt / doc_prompt follow each model card; the instruction only goes on the question side.
MODELS = {
    "qwen3-8b": {
        "id": "Qwen/Qwen3-Embedding-8B",
        "query_prompt": f"Instruct: {TASK}\nQuery: ",
        "doc_prompt": "",
        "dtype": "bfloat16",
    },
    "qwen3-4b": {
        "id": "Qwen/Qwen3-Embedding-4B",
        "query_prompt": f"Instruct: {TASK}\nQuery: ",
        "doc_prompt": "",
        "dtype": "bfloat16",
    },
    "embeddinggemma": {
        "id": "google/embeddinggemma-300m",
        "query_prompt": "task: search result | query: ",
        "doc_prompt": "title: none | text: ",
        # EmbeddingGemma's activations overflow in float16 and it is small, so keep float32.
        "dtype": "float32",
    },
    "bge-m3": {
        "id": "BAAI/bge-m3",
        "query_prompt": "",
        "doc_prompt": "",
        "dtype": "float16",
    },
}
