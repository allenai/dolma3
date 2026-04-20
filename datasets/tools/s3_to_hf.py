#!/usr/bin/env python3
"""Translate an S3 path from a dolma3 midtraining YAML config to its
HuggingFace counterpart in `allenai/dolma3_dolmino_pool`.

Usage:
  python s3_to_hf.py s3://ai2-llm/preprocessed/...     # single path
  echo s3://... | python s3_to_hf.py -                 # stdin
  python s3_to_hf.py --yaml path/to/config.yaml        # every path in a YAML
  python s3_to_hf.py --dump-all [OUT.tsv]              # regenerate the TSV
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_YAML_GLOB = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "configs", "midtraining", "*.yaml")
)
DEFAULT_TSV = os.path.join(SCRIPT_DIR, "s3_to_hf_mapping.tsv")

HF_REPO = "allenai/dolma3_dolmino_pool"
HF_URI_BASE = f"hf://datasets/{HF_REPO}/data"
HF_WEB_BASE = f"https://huggingface.co/datasets/{HF_REPO}/tree/main/data"
PREPROCESSED_PREFIX = "s3://ai2-llm/preprocessed/"

PDF_TOPIC = {
    "art_design": "art_and_design",
    "crime_law": "crime_and_law",
    "education_jobs": "education_and_jobs",
    "entertainment": "entertainment",
    "finance_business": "finance_and_business",
    "hardware": "electronics_and_hardware",
    "health": "health",
    "history": "history_and_geography",
    "home_hobbies": "home_and_hobbies",
    "industrial": "industrial",
    "literature": "literature",
    "politics": "politics",
    "religion": "religion",
    "science_tech": "science_math_and_technology",
    "software": "software",
    "software_dev": "software_development",
    "sports_fitness": "sports_and_fitness",
    "transportation": "transportation",
}

RULES: list[tuple[str, str]] = [
    # Code
    ("stack-edu/sample-fim-weighted-pl-edu-score",  "stack_edu_fim-*"),
    ("stack-edu/fim/documents",                      "stack_edu_fim-*"),
    ("stackedu-fim-20pct-natural",                   "stack_edu_fim-*"),
    ("stack-edu-fim/weighted-pl-20B-v0",             "stack_edu_fim-*"),
    ("tokyotech-llm/swallowcode/scor_final_data",    "cranecode"),
    # Math
    ("megamath_web_pro_max",                         "megamatt"),
    ("tokyotech-llm/swallowmath/beaker_outputs",     "cranemath"),
    ("flat_dolmino_math",                            "dolmino-math"),
    ("dolmino-math-1124-retok",                      "dolmino-math"),
    ("OpenMathReasoning-rewrite-full-thoughts",      "omr-rewrite-fullthoughts"),
    ("tinyMATH/MIND",                                "tinymath-mind"),
    ("tinyMATH/PoT",                                 "tinymath-pot"),
    ("math-meta-reasoning",                          "math-meta-reasoning"),
    ("code-meta-reasoning",                          "code-meta-reasoning"),
    # Verifiable
    ("verifiable/gpt-41",                            "verifiable-gpt41"),
    ("verifiable/o4-mini-high",                      "verifiable-o4mini"),
    ("verifiable-v2/o4-mini-high",                   "verifiable-o4mini"),
    # Reasoning traces
    ("big-reasoning-traces",                         "r1-reasoning-traces"),
    ("qwq-redo-reformatted",                         "qwq-reasoning-traces"),
    ("thinking-data/qwq-traces",                     "qwq-reasoning-traces"),
    ("gemini-redo-reformatted",                      "gemini-reasoning-traces"),
    ("llama-nemotron-processed",                     "llama_nemotron-reasoning-traces"),
    ("openthoughts2-filtered",                       "openthoughts2-reasoning-traces"),
    # QA / instruction / web rewrites
    ("Nemotron-CC/v0/quality=high/kind=synthetic/kind2=diverse_qa_pairs",
                                                     "nemotron-synth-qa"),
    ("tulu-3-sft-for-olmo-3-midtraining",            "tulu-3-sft"),
    ("tulu_flan/v1-FULLDECON-HARD-TRAIN-60M",        "dolmino_1-flan"),
    ("densesub_highthresh",                          "reddit_to_flashcards"),
    ("densesub_lowthresh",                           "reddit_to_flashcards"),
    ("wiki_psgqa_rewrites/psgqa_rewrites_v1",        "wiki_to_rcqa-part{1,2,3}"),
    # STEM-heavy crawl (sponge eli5)
    ("sponge_63_mixes/eli5_60pct_filter",            "stem-heavy-crawl"),
    ("sponge/sponge_63_mixes/eli5_60%_decon_final",  "stem-heavy-crawl"),
    # External datasets (not part of allenai/dolma3_dolmino_pool)
    ("finemath/finemath-3plus",                      "hf://datasets/HuggingFaceTB/finemath/finemath-3plus"),
]

PDF_RE = re.compile(
    r"s2pdf_dedupe_minhash_v1.*?compression(?:-decon)?(?:-2)?/"
    r"length_(?P<length>2e1[23])/(?P<topic>[^/]+)/"
)
WEB_RE = re.compile(
    r"all_dressed_v3_weborganizer_ft_dclm_plus2_vigintiles/"
    r"vigintile_(?P<vig>\d{4})_subset(?:-decon)?(?:-2)?/(?P<topic>[^/]+)/"
)

S3_PATH_RE = re.compile(r"(?:s3|gs)://[^\s'\"#]+")


@dataclass
class Mapping:
    folder: str | None
    note: str = ""

    def hf_uri(self) -> str | None:
        if not self.folder:
            return None
        if self.folder.startswith("hf://"):
            return self.folder
        return f"{HF_URI_BASE}/{self.folder}"

    def hf_url(self) -> str | None:
        if not self.folder:
            return None
        if self.folder.startswith("hf://"):
            tail = self.folder[len("hf://datasets/"):]
            owner, name, *sub = tail.split("/", 2)
            subpath = f"/{sub[0]}" if sub else ""
            return f"https://huggingface.co/datasets/{owner}/{name}/tree/main{subpath}"
        if "*" in self.folder or "{" in self.folder:
            return f"https://huggingface.co/datasets/{HF_REPO}/tree/main/data"
        return f"{HF_WEB_BASE}/{self.folder}"


def translate(s3_path: str) -> Mapping:
    p = s3_path.strip()
    if p.startswith("gs://"):
        p = "s3://" + p[5:]

    m = PDF_RE.search(p)
    if m:
        topic = m.group("topic")
        hf_topic = PDF_TOPIC.get(topic, topic)
        note = "" if topic in PDF_TOPIC else f"unknown PDF topic '{topic}'"
        return Mapping(f"olmocr_science_pdfs-high_quality-{hf_topic}-{m.group('length')}", note)

    m = WEB_RE.search(p)
    if m:
        vig = int(m.group("vig"))
        hf_vig = {18: 19, 20: 20}.get(vig)
        if hf_vig is None:
            return Mapping(None, f"vigintile {vig} has no mapping (only 18→19 and 20)")
        note = "vigintile 18 was renamed to 19" if vig == 18 else ""
        return Mapping(f"common_crawl-high-quality_{hf_vig}_{m.group('topic')}", note)

    for needle, folder in RULES:
        if needle in p:
            return Mapping(folder, _rule_note(folder, p))

    return _lcp_lookup(p) or Mapping(None, "no match")


def _rule_note(folder: str, path: str) -> str:
    if folder == "reddit_to_flashcards":
        if "highthresh" in path:
            return "high_relevance subset"
        if "lowthresh" in path:
            return "low_relevance subset"
    if folder.endswith("*"):
        return "one folder per language × shard"
    if "{" in folder:
        return "split across multiple folders"
    return ""


_CORPUS: list[tuple[str, str, str]] | None = None


def _corpus() -> list[tuple[str, str, str]]:
    global _CORPUS
    if _CORPUS is not None:
        return _CORPUS
    rows: list[tuple[str, str, str]] = []
    if os.path.exists(DEFAULT_TSV):
        with open(DEFAULT_TSV) as fh:
            next(fh, None)
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                parts += [""] * (4 - len(parts))
                if parts[1]:
                    rows.append((parts[0], parts[1], parts[3]))
    _CORPUS = rows
    return rows


def _lcp_lookup(query: str) -> Mapping | None:
    threshold = len(PREPROCESSED_PREFIX)
    best_len = threshold
    best: list[tuple[str, str, str]] = []
    for row in _corpus():
        a, b = query, row[0]
        n = 0
        for i in range(min(len(a), len(b))):
            if a[i] != b[i]:
                break
            n = i + 1
        if n <= threshold:
            continue
        if n > best_len:
            best_len, best = n, [row]
        elif n == best_len:
            best.append(row)
    if not best:
        return None

    retreated = False
    if best_len != len(query) and not query[:best_len].endswith("/"):
        last_slash = query[:best_len].rfind("/")
        if last_slash < threshold:
            return None
        best_len = last_slash + 1
        best = [row for row in _corpus() if row[0].startswith(query[:best_len])]
        retreated = True
        if not best:
            return None

    folders = sorted({row[1].split("/data/", 1)[1] for row in best})
    if len(folders) == 1 and not retreated:
        return Mapping(folders[0], best[0][2])
    return Mapping(None, f"ambiguous; candidates: {', '.join(folders)}")


def extract_s3_paths(text: str) -> set[str]:
    return {
        p
        for line in text.splitlines()
        if not line.lstrip().startswith("#")
        for p in S3_PATH_RE.findall(line)
    }


def dump_all(yaml_glob: str, out_path: str) -> int:
    paths: set[str] = set()
    files = sorted(glob.glob(yaml_glob))
    for f in files:
        with open(f) as fh:
            paths.update(extract_s3_paths(fh.read()))
    _corpus()  # warm the cache before we truncate out_path (may be DEFAULT_TSV)
    with open(out_path, "w") as out:
        out.write("s3_path\thf_uri\thf_url\tnote\n")
        for s3 in sorted(paths):
            m = translate(s3)
            out.write("\t".join([s3, m.hf_uri() or "", m.hf_url() or "", m.note]) + "\n")
    print(f"wrote {len(paths)} paths from {len(files)} YAML(s) to {out_path}", file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("path", nargs="?", help="S3 path, or '-' to read from stdin")
    ap.add_argument("--yaml", help="translate every s3:// path in a YAML config")
    ap.add_argument("--url", action="store_true",
                    help="print only the clickable https:// URL (single-path mode)")
    ap.add_argument("--dump-all", nargs="?", const=DEFAULT_TSV, metavar="OUT.tsv",
                    help=f"walk --yaml-glob and write mappings to TSV (default: {DEFAULT_TSV})")
    ap.add_argument("--yaml-glob", default=DEFAULT_YAML_GLOB,
                    help="glob for --dump-all (default: %(default)s)")
    args = ap.parse_args()

    if args.dump_all:
        return dump_all(args.yaml_glob, args.dump_all)

    if args.yaml:
        with open(args.yaml) as fh:
            paths = sorted(extract_s3_paths(fh.read()))
        for s3 in paths:
            m = translate(s3)
            tail = f"  # {m.note}" if m.note else ""
            print(f"{s3}\n  → {m.hf_uri() or '(no mapping)'}{tail}")
            if m.hf_url():
                print(f"     {m.hf_url()}")
        return 0

    if args.path is None:
        ap.print_help()
        return 1

    s3 = sys.stdin.read().strip() if args.path == "-" else args.path
    m = translate(s3)
    if m.folder:
        if args.url:
            print(m.hf_url())
        else:
            print(m.hf_uri())
            if m.hf_url():
                print(m.hf_url())
        if m.note:
            print(f"# {m.note}", file=sys.stderr)
        return 0
    print(f"# {m.note}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
