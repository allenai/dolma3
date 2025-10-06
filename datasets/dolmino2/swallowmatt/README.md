# SwallowMatt Data

This outlines the steps taken to create the SwallowMatt dataset. This can be thought of as a more permissively licensed variation on the [SwallowMath dataset](https://huggingface.co/datasets/tokyotech-llm/swallow-math) from the `tokyotoch-llm` team. SwallowMath leveraged Llama-3.3-70B-Instruct to generate data, which has a less permissive license. By contrast, we used Qwen3-32B.

## Overview
This is a simple one-step rewrite of an existing permissively licensed dataset: we simply use Qwen3 to rewrite Finemath-4plus.

## Source Materials

- **Base Dataset**: [Finemath-4plus](https://huggingface.co/datasets/HuggingFaceTB/finemath) (6,699,493 Training examples)
- **Inspiration**: [SwallowMath](https://arxiv.org/pdf/2505.02881) Paper outlining the data generation procedure


## Recipe
Starting with Finemath-4plus, for each example, we use Qwen3-32B to rewrite the example by prepending with a system message containing the prompt:
```
You are an intelligent math tutor. You are given the following math problem and answer with some unnecessary parts. Please remove the unneeded parts of the questions. For example, the date of the question submitted, the answer date, the privacy policy, the footer, the header, etc, should be removed. However, please keep the main question and answer.\nIf questions or answers lack some information or are not elaborate, please make them more informative and easy to understand. If needed, please add more detail about the step-by-step calculation process.\n\nHere is the example:
```


## Final Dataset
Ultimately this yields 6,553,181 documents (one for each fm4+ example, minus some generation failures), for a total of 5.625B tokens.
