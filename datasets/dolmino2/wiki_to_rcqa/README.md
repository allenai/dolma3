## Overview

This folder contains supplementary documentation of **wiki-to-rcqa**. This dataset consists of wikipedia passages concatenated with synthetically-generated QA pairs about the content of those passages.

Supplementary files in this folder:

- Prompt templates for synthetic QA pair generation: `prompts-wiki-to-rcqa-py`

Below we elaborate on the data creation process.

## Data creation steps

### 1. Wikipedia passage extraction

Initial Wikipedia preprocessing used [wikiclean.py](https://github.com/allenai/dolma/blob/soldni/wpp/contrib/wikiclean/wikiclean.py). This script aims to identify Wikipedia sections.

The outputs of this processing were then subjected to further segmentation to control passage length. Segmentation was executed according to the following constraints:

- if section is less than 300 words, include as single passage
- if section is >= 300 words, split into paragraphs and include paragraphs as passages
- omit passages < 20 words

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
qnum_prop = round(len(passage.split())/40)
if qnum_prop < 2:
    qnum = 1
else:
    qnum = max(1,min(8,random.choices(range(qnum_prop-4,qnum_prop))[0]))
```



