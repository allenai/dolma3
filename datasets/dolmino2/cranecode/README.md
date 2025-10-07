# CraneCode Data

This outlines the steps taken to create the CraneCode dataset. This can be thought of as a more permissively licensed variation on the [SwallowCode dataset](https://huggingface.co/datasets/tokyotech-llm/swallow-math) from the `tokyotoch-llm` team. SwallowCode leveraged Llama-3.3-70B-Instruct to generate data, which has a less permissive license. By contrast, we used Qwen2.5-Coder-32B-Instruct.

## Overview
There are many steps to reproduce SwallowCode: first the input data needs to be downloaded from The-Stack-v2-smol, and selected just for python data. Then we filter to keep only compilable code segments that have a linting score >= 7.0 (according to the SwallowCode linting + scoring guidelines). Then we apply two rounds of rewrites using Qwen2.5-Coder-32B-Instruct: the first aims to coerce the code into proper style, while the second ensures the code is self-contained and optimized.

## Source Materials

- **Base Dataset**: [The-Stack-v2-train-smol (Python only)](https://huggingface.co/datasets/bigcode/the-stack-v2-train-smol-ids) (41,025,639 Training Examples)
- **Inspiration**: [SwallowCode](https://arxiv.org/pdf/2505.02881) Paper outlining the data generation procedure


## Recipe
We start with just the python subset of The-Stack-v2-smol. The first step is to check if each python string passes the python `compile` command, and if so, we use pylint to score the code. We apply a comment penalty similar to SwallowCode and then keep only documents with a score of at least 7.0. This yields a dataset of 20,424,729 documents.

Then we rewrite in order to attain python code conforming to style guidelines. Following SwallowCode, we rewrite each example by prepenting with a system message containing the prompt:
```
You are a smart software engineer. Please evaluate the following code on a scale of 1 to 10 based on the following criteria:\n\n1. Are variable names descriptive and consistent with naming conventions?\n2. Are comments and doc-strings appropriately written to explain the purpose and functionality of the code?\n3. Are type annotations used effectively where applicable?\n4. Are functions appropriately modularized, with well-defined responsibilities and clear separation of concerns?\n5. Are variables' lifetimes intentionally managed, avoiding frequent reassignment or overly long scopes?\n6. Is error handling implemented appropriately where necessary?\n7. Is the code properly indented and follows standard formatting guidelines?\n8. Do comments provide context and rationale, rather than merely describing what the code does?\n9. Are functions and classes designed with clear, single responsibilities?\n10. Is the code formatted in a way that enhances readability?\n\n\nAnd provide suggestions for improvement based on the evaluation criteria. You can also provide an improved version of the code like the following style:\n\n### Evaluation: 7\n\n\n### Suggestions:\n\n    Provide specific, actionable suggestions to improve the code based on the evaluation criteria.\n\n\n### Improved Code:\n\nProvide a revised version of the code incorporating the suggested improvements.\n\n```python\n\ndef improved_function(arg1: int, arg2: str) -> str:\n    # Your improved code here\n    pass\n```\n\n\n
```

We then run a filtering step just to select out the code segment of the output. This provides us with 19,754,951 style-rewritten documents. 

Then we apply the optimization-rewrite, by prepending each example with a system message containing the prompt:
```
You are a smart software engineer. Please change a given code into self-contained and well-structured code following the below best practices and pythonic way.\n1. Use meaningful variable and function names.\n2. Write a clear and concise docstring for the function.\n3. Use type hints for the function signature.\n4. Write a clear and concise comment for the code block.\n5. Ensure the code is self-contained and does not depend on external variables.\n6. Ensure the code is well-structured and easy to read.\n7. Ensure the code is free of errors and runs correctly.\n8. Ensure the code is optimized and does not have redundant operations.\n9. Ensure the algorithm and data structures are efficient and concise.\n\nIf given code is not self-contained or too simple, please change it to a more educational and useful code.\n
```

Then we once again run a filtering step just to select out the code segment of the output.

## Final Dataset
Ulitmately this yields 19,667,524 documents, for a total of 18.83B tokens. 