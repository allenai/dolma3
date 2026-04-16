# tools

## `s3_to_hf.py`

Translate an S3 path from a dolma3 midtraining YAML config into its
HuggingFace counterpart in
[`allenai/dolma3_dolmino_pool`](https://huggingface.co/datasets/allenai/dolma3_dolmino_pool).

Useful if you want to replicate or inspect the data referenced by the
configs under [`datasets/configs/midtraining/`](../configs/midtraining/)
but don't have access to the internal S3 bucket.

### Usage

```bash
# Single path
python s3_to_hf.py s3://ai2-llm/preprocessed/.../*.npy

# Just the clickable URL
python s3_to_hf.py --url s3://ai2-llm/preprocessed/.../*.npy

# From stdin
echo s3://... | python s3_to_hf.py -

# Every path in a YAML config
python s3_to_hf.py --yaml ../configs/midtraining/anneal-round5-olmo3_7b-anneal-decon-12T.yaml

# Regenerate s3_to_hf_mapping.tsv from every midtraining YAML
python s3_to_hf.py --dump-all
```

Partial paths work too — pass anything from a dataset root down to a
specific shard and the tool returns the best matching HF folder, or a
list of candidates when the prefix is ambiguous:

```bash
python s3_to_hf.py s3://ai2-llm/preprocessed/sponge_63_mixes/
# → hf://datasets/allenai/dolma3_dolmino_pool/data/stem-heavy-crawl

python s3_to_hf.py s3://ai2-llm/preprocessed/thinking-data/
# # ambiguous; candidates: gemini-reasoning-traces, llama_nemotron-reasoning-traces, ...
```

### `s3_to_hf_mapping.tsv`

Static lookup of every S3 path referenced in the midtraining YAMLs →
its HF URI, browser URL, and any notes (e.g. `vigintile 18 was renamed
to 19`). Regenerate with `--dump-all` whenever the YAMLs change.
