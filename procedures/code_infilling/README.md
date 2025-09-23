## Code Infilling

### Summary

Code infilling, or code FIM (fill-in-middle), is a task where a model is given a piece of code with a segment missing, and it is required to generate that segment based on the surrounding context. This task is particularly useful for scenarios such as code completion, bug fixing, and code synthesis.

Our implementation follows the infilling procedure from the paper [StarCoder2](https://arxiv.org/pdf/2402.19173) in which we split the code documents into 3 segments: a prefix, a middle, and a suffix. In our final midtraining mixture we performed FIM transformation on 50% of the documents and left the other 50% unchanged. In addition to the standard prefix-suffix-middle transformations we also performed a suffix-prefix-middle arrangement on 50% of the transformed documents.

### Implementation Details

The FIM transformation includes the following steps:

1. **Selection**: We select code documents randomly based on a specified FIM rate (e.g., 50% of the documents).

2. **Segmentation**: Each document is split into three segments: a prefix, a middle, and a suffix.

3. **Transformation**: We apply the FIM transformation to the selected documents by injecting sentinel tokens around the segments and arranging based on whether the document should be a prefix-suffix-middle or suffix-prefix-middle arrangement.

4. **Tokenization**: The transformed documents are tokenized using a tokenizer compatible with the model being trained. In our case we used the [dolma2-tokenizer](https://huggingface.co/allenai/dolma2-tokenizer/tree/main)

### Tooling

We provide a set of tools to replicate the FIM transformation process. These tools are available in the [Dolma repository](https://github.com/allenai/dolma) and requires a valid Rust installation to build and run.

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

**Tokenization**
```yaml
documents:
- input_dir/*.jsonl.zst
destination: output_dir
seed: 3920
max_size: 536870912
processes: 128
dtype: uint32
tokenizer:
  name_or_path: allenai/dolma2-tokenizer
  bos_token_id: null
  eos_token_id: 100257
  pad_token_id: 100277
  segment_before_tokenization: false
```

```sh
python dolma -c config.yaml tokens
```
