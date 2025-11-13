## Overview

This folder contains supplementary documentation of **wiki-to-rcqa**. This dataset consists of wikipedia passages concatenated with synthetically-generated QA pairs about the content of those passages.

Supplementary files in this folder:

- Prompt templates for synthetic QA pair generation: `prompts-wiki-to-rcqa-py`
- Wikipedia preprocessing script: `scripts/wikiclean.py`

Below we elaborate on the data creation process.

## Data creation steps

### 1. Wikipedia passage extraction

We performed initial preprocessing to clean the Wikipedia data and format into summaries and sections. The script used for this initial preprocessing (which assumes input data format as produced via the [Wikipedia preparation found here](https://github.com/allenai/dolma/blob/main/docs/getting-started.md)) is provided at `scripts/wikiclean.py`. Example usage for `wikiclean.py` is below: 

```
uv run scripts/wikiclean.py --documents <wikipedia_input_dir> --destination <output_dir> --processes <num_processes>
```

The outputs of this preprocessing were then subjected to further segmentation to control passage length. Segmentation was executed according to the following constraints:

- if section is less than 300 words, include as single passage
- if section is >= 300 words, split into paragraphs (splitting on single newline) and include paragraphs as passages
- omit passages that are < 20 words

### 2. QA generation prompting

We created four prompt templates inspired by instructions given to annotators when generating QA pairs for popular reading comprehension QA benchmarks. Our prompt templates can be found in `prompts-wiki-to-rcqa-py`.

For data generation, we iterated over passages and sampled prompt templates according to the following sampling distribution:

```
distribution = [
        (DEFAULT,.1),
        (SPAN,.25),
        (PPHRASE,.25),
        (DROP,.4),
    ]
```

Each template contains a slot to request a specific number *n* of questions. We select this value from the range [1,8] with a sampling function sensitive to passage length in words, as in the code snippet below:

```
length_scaler = round(len(passage.split())/40)
if length_scaler < 2:
    num_questions = 1
else:
    num_questions = max(1,min(8,random.choices(range(length_scaler-4,length_scaler))[0]))
```



