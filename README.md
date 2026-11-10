# cq-class-retrieval

Given a competency question (CQ), which ontology classes do you need to answer it? This project ranks the 44 classes of the Enslaved Ontology for each CQ by embedding similarity and measures how often the gold classes end up near the top.

It is the retriever stage of a thesis project. It compares four embedding models and six ways of turning a class into text ("views"). Only classes are retrieved, not properties. There is no reranker or LLM judge yet; those are planned as separate stages that take this ranking as input.

## How it works

Every class is written out as text in each view, embedded once per model, and compared with the embedded CQ by cosine similarity. The classes are then sorted by score, and the ranking is scored against the gold classes listed for that CQ.

The pipeline has four stages, each in its own folder:

| stage | what it does | needs a GPU |
|---|---|---|
| `prepare` | reads the class JSON and the CQ csv, builds the views, checks that every gold class exists | no |
| `embed` | encodes the CQs and every class view with each model, saves `.npy` files | yes |
| `rank` | ranks all 44 classes per CQ, per model and view | no |
| `evaluate` | computes recall, precision, R-precision and MRR, plus a random baseline | no |

Only `embed` imports torch and sentence-transformers, so the other three stages run on any machine.

## Data

Both inputs live in `data/`.

`class_descriptions.json` is a list of 44 classes. Each has a `name` (such as `AgeRecord`), a human-readable `label` ("Age Record"), a `description`, and three lists of class names: `parent`, `children` and `neighbours`. Neighbours are the other classes this one is linked to in the ontology; a few of them are also its parent or child.

`cqs.csv` have competency questions with columns `cq_id`, `question` and `gold_classes`, the last one separated by `|`. Each CQ has between 6 and 13 gold classes.

## Class views

A view is the text that stands in for a class when it is embedded. The CQ side is always the plain question.

| view | text |
|---|---|
| `label` | the label alone: "Age Record" |
| `description` | the description alone |
| `label_desc` | "Age Record. A record of a person's age..." |
| `label_desc_hier` | `label_desc` plus "Parent: Agent Record." and "Subclasses: ..." where they exist |
| `description_filtered` | the description with function words and punctuation removed |
| `label_desc_hier_neigh` | `label_desc_hier` plus "Neighbour entities: ..." listing the neighbours |

Parents, children and neighbours are written with their labels, not their class names. A neighbour that is already a parent or child is left out of "Neighbour entities:" so no label appears twice. Classes with no hierarchy or neighbours get no suffix, so for them the longer views equal `label_desc`.

`description_filtered` removes prepositions, articles, wh-words, auxiliary and modal verbs, conjunctions and pronouns, using fixed word lists at the top of `prepare/prepare.py`. Matching is whole-word and ignores case, multi-word phrases like "as long as" go as a unit, and hyphenated or possessive words (`top-level`, `person's`) are kept intact. Content verbs such as "used" stay, because a word list cannot tell the verb "record" from the noun. The Agent description, for example, becomes:

> person organization enslaved individual owner enslaver institution top-level class entities bear agent records ontology subclasses Person Organization

That cuts the descriptions from 1,560 words to 935 in total.

## Models

| key | model | dtype | prompts |
|---|---|---|---|
| `qwen3-8b` | Qwen/Qwen3-Embedding-8B | bfloat16 | query: `Instruct: {task}\nQuery: ` |
| `qwen3-4b` | Qwen/Qwen3-Embedding-4B | bfloat16 | query: `Instruct: {task}\nQuery: ` |
| `embeddinggemma` | google/embeddinggemma-300m | float32 | query: `task: search result \| query: `, class: `title: none \| text: ` |
| `bge-m3` | BAAI/bge-m3 | float16 | none |

The task text is "Given a competency question, retrieve the ontology classes needed to answer it". Prompts follow each model card, and the instruction only goes on the question side. EmbeddingGemma runs in float32 because its activations overflow in float16.

Vectors are normalised to unit length in float32 after encoding, so a dot product is the cosine similarity. Normalising in bfloat16 left the lengths between 0.998 and 1.002, which the save step rejects.

## Setup

```bash
pip install torch sentence-transformers numpy
```

Pick the torch build that matches your GPU from the PyTorch install page. A few combinations that have worked:

- RTX 4070 laptop (8 GB): torch 2.15.0+cu132, sentence-transformers 6.2.0, transformers 5.22.1, numpy 1.26.4. Fits bge-m3 and EmbeddingGemma, not the Qwen models.
- Tesla V100 16 GB, Python 3.10: torch 2.14.1+cu126, sentence-transformers 6.2.0, transformers 5.20.2. Runs both Qwen models. CUDA 13 builds of torch have no kernels for the V100, so use a cu12x build there.

EmbeddingGemma is gated on Hugging Face: accept its licence, then run `hf auth login`. The Qwen models and bge-m3 are not gated.

If you see `OMP: Error #15 ... libiomp5md.dll already initialized` on Windows, reinstall numpy in that environment with `pip install --force-reinstall numpy`.

## Usage

Run everything from the project folder:

```bash
python main.py prepare
python main.py embed --models bge-m3 embeddinggemma
python main.py rank
python main.py evaluate
```

`python main.py all` runs the four stages in order. It stops if `prepare` fails, but a model that fails to embed does not stop the others.

Options shared by every stage:

| option | default |
|---|---|
| `--classes-path` | `data/class_descriptions.json` |
| `--cqs-path` | `data/cqs.csv` |
| `--out-dir` | `out/cq_class_retrieval` |
| `--models` | all four model keys |
| `--no-instruction` | off; embeds with an empty query prompt and adds `__noinstr` to the run name |
| `--self-test` | with `evaluate`: checks the metric code on perfect and reversed rankings, no embeddings needed |

`embed`, `rank` and `evaluate` skip a model that fails or has no embeddings, report it at the end and exit with code 1. `evaluate` picks up existing `__noinstr` runs automatically and puts them in the same table.

### Embedding on another machine

The models can be embedded wherever the GPU is and copied back. Copy the project folder (including `data/`) to the GPU machine, run `prepare` and then `embed` for the models you need there, and copy each `out/cq_class_retrieval/embeddings/<model>/` folder back to the same place locally. `rank` checks that the embeddings were made from the same classes and CQs as the local `classes.csv` and `cqs.csv`, and skips the run with a message if they differ.

`embed` always encodes every view of a model. After adding a view, re-embed all models, or `rank` will skip them for the missing file.

## Outputs

Everything goes under `--out-dir`:

```
classes.csv                           one row per class, one column per view
cqs.csv                               cq_id, question, gold_classes
embeddings/<run>/cqs.npy              one unit vector per CQ
embeddings/<run>/classes__<view>.npy  one unit vector per class
embeddings/<run>/meta.json            model, dtype, prompts, row order
rankings/<run>__<view>.csv            all 44 classes ranked for each CQ
results/summary.csv                   mean metrics per run and view
results/per_cq.csv                    metrics per CQ, with the gold classes missed in the top 10
```

## Layout

```
main.py                 CLI entry point
common/config.py        paths, VIEWS, task text, MODELS
common/files.py         csv helpers and run naming
prepare/prepare.py      load and validate inputs, build the views, stop-word lists
embed/embed.py          load models, encode, save embeddings
rank/rank.py            cosine ranking
evaluate/evaluate.py    metrics, random baseline, self-test
data/                   class_descriptions.json, cqs.csv
```

Each stage folder exports a `run_<stage>` function from its `__init__.py`. Imports are top level (`from common.config import ...`), which works because Python puts the folder containing `main.py` on the import path.

## Next steps

- Add a reranker over the top-20 shortlist, and possibly query splitting or rank fusion.
