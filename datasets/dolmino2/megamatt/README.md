# MegaMatt Data
This outlines the steps taken to create the MegaMatt dataset. MegaMatt can be thought of as a more permissively licensed variation on the Megamath-Web-Pro-Max dataset, fom the GAIR team in the [OctoThinker project](https://arxiv.org/pdf/2506.20512). Megamath-Web-Pro-Max leveraged Llama-3.1-70B to generate data, which has a less permissive license. By contrast, for MegaMatt we used Qwen3-32B for all data generation. 

## Overview
Creation of Megamatt involves a simple one-step rewrite of an existing permissively licensed dataset: we simply used Qwen3 to rewrite a subset of CommonCrawl/MegaMath-Web-Pro, adhering to the prompt used by the GAIR team to create Megamath-Web-Pro-Max.

## Source Materials
- **Base Dataset:** We use data from [Megamath-Web-Pro](https://huggingface.co/datasets/LLM360/MegaMath), using only data from CommonCrawl dumps taken from dump CC-MAIN-2023-23 and later.
- **Inspiration:** We refine webtext using a prompt from the [OctoThinker project](https://arxiv.org/pdf/2506.20512)



## Recipe
We first collect data from [Megamath-Web-Pro](https://huggingface.co/datasets/LLM360/MegaMath) and then filter down to only collect documents that were taken from Commoncrawl dumps CC-MAIN-2023-23 and later. This preserves 7,216,941 documents needing refinement. Then we use Qwen3-32B and the following prompt to refine them:
```
Task:\n- Carefully analyze the provided text to extract key facts, concrete details, important numbers,\nand core concepts.\n- Remove any irrelevant or noisy information, and reorganize the content into a logically structured,\ninformation-dense, and concise version that is easy to learn from. Output only the refined text.\n- Strive to maintain the original length as much as possible (avoid excessive shortening).\n- Refine multiple choice questions and answers if any.\nText:\n
```

## Final Dataset:
Ultimately this yields 6,794,731 refined documents that (one for each of the Megamath-Web-Pro) 3.884B tokens.
