"""Load class_descriptions.json and the CQ csv, build the class text views and validate the gold classes."""

import csv
import json
import os
import re
import sys

from common.files import write_csv

# Function words removed for the description_filtered view. Each list starts with the original
# words; the line after "# added" extends it within the same word class.
PREPOSITIONS = [
    "in", "on", "at", "by", "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "of", "up", "down", "over", "under",
    # added
    "across", "along", "among", "around", "as", "behind", "beside", "beyond", "despite", "except",
    "near", "off", "onto", "per", "than", "toward", "towards", "upon", "via", "within", "without",
    "unlike", "throughout",
]
ARTICLES = ["a", "an", "the"]
WH_QUESTIONS = [
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    # added
    "whatever", "whichever", "whoever", "wherever",
]
# Auxiliary and modal verbs only: a word list can't tell content verbs ("record") from nouns.
VERBS = [
    "is", "are", "was", "were", "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should",
    # added
    "am", "be", "been", "being", "having", "can", "cannot", "may", "might", "must", "shall",
]
CONJUNCTIONS = [
    "and", "but", "or", "nor", "for", "yet", "so", "because", "although", "since", "unless", "while",
    "whereas", "after", "before", "once", "until", "when", "whenever", "if", "provided that",
    "as long as", "in case",
    # added
    "though", "even though", "whether", "both", "either", "neither", "as well as", "rather than",
    "so that",
]
PRONOUNS = [
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "he", "him", "his",
    "she", "her", "hers", "this", "that", "these", "those", "such",
]

STOP_WORDS = sorted(set(PREPOSITIONS + ARTICLES + WH_QUESTIONS + VERBS + CONJUNCTIONS + PRONOUNS))
# Longest first so "as long as" is removed whole before "as"; the lookarounds treat hyphens and
# apostrophes as part of a word, so "add-on" and "person's" survive and class names never match.
STOP_ALTERNATIVES = "|".join(re.escape(word) for word in sorted(STOP_WORDS, key=len, reverse=True))
STOP_PATTERN = re.compile(rf"(?<![\w'-])(?:{STOP_ALTERNATIVES})(?![\w'-])", re.IGNORECASE)


# Any punctuation, except a hyphen or apostrophe with a word character on both sides: those belong
# to the word ("top-level", "person's"), not to the sentence. "_" is listed because \w includes it.
PUNCTUATION_PATTERN = re.compile(r"[^\w\s'-]|_|(?<!\w)['-]|['-](?!\w)")


def filter_description(text):
    """Remove every STOP_WORDS entry (whole words, any case) and all punctuation from `text`."""
    # Single spaces first, so phrases like "as long as" match across line breaks or double spaces.
    text = " ".join(text.split())
    text = STOP_PATTERN.sub(" ", text)
    text = PUNCTUATION_PATTERN.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_classes(path):
    """Read class_descriptions.json and check every class has the expected keys and a unique name."""
    if not os.path.isfile(path):
        sys.exit(f"error: {path} not found")
    with open(path, encoding="utf-8") as handle:
        classes = json.load(handle)
    if not isinstance(classes, list):
        sys.exit(f"error: {path} should contain a list of class objects")

    required = {"name", "label", "description", "parent", "children", "neighbours"}
    seen = set()
    for position, cls in enumerate(classes):
        missing = required - set(cls)
        if missing:
            sys.exit(f"error: class #{position} ({cls.get('name', '?')}) is missing {sorted(missing)}")
        if not all(isinstance(cls[field], list) for field in ("parent", "children", "neighbours")):
            sys.exit(f"error: class {cls['name']!r}: 'parent', 'children' and 'neighbours' must be lists")
        if cls["name"] in seen:
            sys.exit(f"error: duplicate class name {cls['name']!r}")
        seen.add(cls["name"])
    return classes


def build_class_views(classes):
    """Turn each class into one row: its name plus the text of every view in VIEWS (common/config.py)."""
    label_of = {cls["name"]: cls["label"] for cls in classes}

    def labels(cls, field, skip=()):
        unknown = [name for name in cls[field] if name not in label_of]
        if unknown:
            sys.exit(f"error: class {cls['name']!r} has unknown {field} {unknown}")
        return ", ".join(label_of[name] for name in cls[field] if name not in skip)

    rows = []
    for cls in classes:
        label_desc = f"{cls['label']}. {cls['description']}"
        hierarchy = []
        if cls["parent"]:
            hierarchy.append(f"Parent: {labels(cls, 'parent')}.")
        if cls["children"]:
            hierarchy.append(f"Subclasses: {labels(cls, 'children')}.")
        # Neighbours that are already a parent or child would repeat a label the hierarchy part names.
        related = labels(cls, "neighbours", skip=cls["parent"] + cls["children"])
        neighbourhood = [f"Neighbour entities: {related}."] if related else []
        rows.append({
            "name": cls["name"],
            "label": cls["label"],
            "description": cls["description"],
            "label_desc": label_desc,
            # Classes with no parent or children get no suffix, so this view equals label_desc for them.
            "label_desc_hier": " ".join([label_desc] + hierarchy),
            "description_filtered": filter_description(cls["description"]),
            "label_desc_hier_neigh": " ".join([label_desc] + hierarchy + neighbourhood),
        })
    return rows


def load_cqs(path):
    """Read the CQ csv, keeping cq_id, question and gold_classes (split on '|'); gold_properties is ignored."""
    if not os.path.isfile(path):
        sys.exit(f"error: {path} not found")
    # utf-8-sig drops a byte-order mark if the file was saved from Excel; otherwise the first column reads as '﻿cq_id'.
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"cq_id", "question", "gold_classes"} - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"error: {path} is missing columns {sorted(missing)}")
        rows = list(reader)

    cqs, seen = [], set()
    for row in rows:
        cq_id = row["cq_id"].strip()
        gold = [name.strip() for name in row["gold_classes"].split("|") if name.strip()]
        if cq_id in seen:
            sys.exit(f"error: duplicate cq_id {cq_id!r} in {path}")
        if not row["question"].strip() or not gold:
            sys.exit(f"error: {cq_id} has an empty question or no gold classes")
        seen.add(cq_id)
        cqs.append({"cq_id": cq_id, "question": row["question"].strip(), "gold_classes": gold})
    return cqs


def validate_gold_classes(class_rows, cqs):
    """List gold classes that are not in the class JSON or appear twice in one CQ; empty means all good."""
    known = {row["name"] for row in class_rows}
    problems = []
    for cq in cqs:
        seen = set()
        for name in cq["gold_classes"]:
            if name not in known:
                problems.append(f"{cq['cq_id']}: gold class {name!r} is not in the class JSON")
            elif name in seen:
                problems.append(f"{cq['cq_id']}: gold class {name!r} is listed more than once")
            seen.add(name)
    return problems


def run_prepare(args):
    """Load and validate both inputs, then write classes.csv and cqs.csv to the output folder."""
    class_rows = build_class_views(load_classes(args.classes_path))
    cqs = load_cqs(args.cqs_path)

    problems = validate_gold_classes(class_rows, cqs)
    if problems:
        print(f"{len(problems)} gold-class problem(s); nothing written:")
        for problem in problems:
            print(f"  {problem}")
        return 1

    classes_path = os.path.join(args.out_dir, "classes.csv")
    cqs_path = os.path.join(args.out_dir, "cqs.csv")
    write_csv(classes_path, class_rows, list(class_rows[0]))
    write_csv(cqs_path,
              [{**cq, "gold_classes": "|".join(cq["gold_classes"])} for cq in cqs],
              ["cq_id", "question", "gold_classes"])

    gold_sizes = [len(cq["gold_classes"]) for cq in cqs]
    print(f"classes: {len(class_rows)}")
    print(f"CQs:     {len(cqs)}")
    print("gold classes per CQ: " + ", ".join(f"{cq['cq_id']}={size}" for cq, size in zip(cqs, gold_sizes)))
    print(f"  mean {sum(gold_sizes) / len(gold_sizes):.1f}, min {min(gold_sizes)}, max {max(gold_sizes)}")
    print("missing gold classes: 0")
    print(f"wrote {classes_path}\n      {cqs_path}")
    return 0
