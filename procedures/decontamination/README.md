## Decontamination

### Summary

Decontamination identifies and removes training documents that contain evaluation data to ensure benchmark scores accurately reflect model capabilities rather than memorization.

Our implementation uses [Decon](https://github.com/allenai/decon), which employs sampled n-gram matching with bidirectional cluster expansion to efficiently detect contamination in large datasets. We removed training documents contaminated with text from evaluation benchmarks including MMLU, GSM8K, ARC, and [others](https://github.com/allenai/decon/blob/main/config/evals.yaml).

### Implementation Details

The decontamination process includes the following steps:

1. **Index Building**: Evaluation datasets are normalized into question, answer, and passage components. N-grams are extracted and indexed with IDF weighting to prioritize distinctive terms.

2. **Detection**: Training documents are sampled periodically for n-gram matches. When a match is found, the algorithm expands bidirectionally to identify the full contaminated span.

3. **Scoring**: Contamination is scored using IDF-weighted n-gram overlap, with adaptive weighting across question (highest priority), answer, and passage components when present.

For instances ≥ 50 tokens (cumulative of question/answer/passage), a perfect question match (IDF overlap score 1.0) alone is sufficient for contamination detection in question-only scenarios. But when answers or passages are present, they serve as supporting evidence of contamination, allowing for more imperfect question matches to still exceed the 0.8 threshold. Below 50 tokens, the system demands increasingly perfect matches as length decreases (reaching 1.0 requirement at ≤20 tokens), making even high-quality matches insufficient without near-perfect alignment of all available components. See [SIMPLE documentation](https://github.com/allenai/decon/blob/main/doc/simple-details.md) for more details.

Documents exceeding the contamination threshold are flagged for removal.

For example, given a training document:
```
...for θ 30 c i θ i0 4 for θ 90 d i θ is constant for all values of θ **the plane face of plano convex lens of focal
length 20 cm is silvered this combination is equivalent to the type of mirror and its focal length is** a convex f 20 c
m b **concave f** 20 cm in a displacement method using convex lens two images are obtained for a separation of d between...
```

And an evaluation question:
```
PROMPT: the plane face of plano convex lens of focal length 20 cm is silvered this combination is equivalent to the type of mirror and its focal length is
ANSWER: concave f 10 cm
```

The training document would be flagged as contaminated and removed from the dataset.

For a detailed technical description of the algorithm, see the [SIMPLE documentation](https://github.com/allenai/decon/blob/main/doc/simple.md).

### Tooling

We provide a Rust utility for decontamination available in the [Decon repository](https://github.com/allenai/decon). This tool requires a valid Rust installation to build and run.

**Contamination Detection**: A command to detect and remove contaminated documents.
```sh
cargo run --release -- detect \
    --training-dir /path/to/training/data \
    --evals-dir /path/to/eval/references \
    --purify
```
