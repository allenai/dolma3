## Code Infilling

### Summary

Code infilling, or code FIM (fill-in-middle), is a task where a model is given a piece of code with a segment missing, and it is required to generate that segment based on the surrounding context. This task is particularly useful for scenarios such as code completion, bug fixing, and code synthesis.

Our implementation follows the infilling procedure from the paper [StarCoder2](https://arxiv.org/pdf/2402.19173) in which we split the code documents into 3 segments: a prefix, a middle, and a suffix. In our final midtraining mixture we performed FIM transformation on 50% of the documents and left the other 50% unchanged. In addition to the standard prefix-suffix-middle transformations we also performed a suffix-prefix-middle arrangement on 50% of the transformed documents.

### Implementation Details

The FIM transformation includes the following steps:

1. **Selection**: We select code documents at random based on a specified FIM rate (e.g., 50% of the documents).

2. **Segmentation**: Each document is split into three segments: a prefix, a middle, and a suffix.

3. **Transformation**: We apply the FIM transformation to the selected documents by injecting sentinel tokens around the segments and arranging based on whether the document should be a prefix-suffix-middle or suffix-prefix-middle arrangement.

For example, given a code document:
```python
def add_two_numbers(a: int, b: int) -> int:
    sum = a + b
    return sum
```

And after transformation with prefix-suffix-middle arrangement, it would look like:
```python
<|fim_prefix>def add_two_numbers(a: int, b: int) -> int:
<|fim_suffix|>
    return sum<|fim_middle|>    sum = a + b
```

### Tooling

We provide a Rust utility to replicate the FIM transformation. This tool is available in the [Dolma repository](https://github.com/allenai/dolma/contrib/fill-in-middle/) and requires a valid Rust installation to build and run.

**FIM Transformation**: A script to perform the FIM transformation on code documents.
```sh
cargo run --release -- \
    --inputs input_dir \
    --output output_dir \
    --fim-rate 5.0 \
    --psm-spm-split 0.5 \
    --fim-prefix-token '<|fim_prefix|>' \
    --fim-middle-token '<|fim_middle|>' \
    --fim-suffix-token '<|fim_suffix|>'
```
