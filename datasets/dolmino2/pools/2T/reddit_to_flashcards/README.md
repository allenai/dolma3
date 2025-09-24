## Overview

This folder contains supplementary documentation of the following sources:
- **reddit_to_flashcards-high_relevance**
- **reddit_to_flashcards-low_relevance**

These datasets are very similar in their creation, differing primarily in that a) they draw on content from different subreddits, and b) they use slightly different formatting in few-shot examples used for prompting for synthetic data generation.

We first cover the shared processing steps involved in creation of both of these datasets, before outlining the specific properties in which they differ.

## Shared processing steps

Both **reddit_to_flashcards-high_relevance** and **reddit_to_flashcards-low_relevance** have the following properties:

### 1. Preliminary filtering

Both sets were derived from a single initial dataset of Reddit submission/comment pairs,  derived from the PushShift Reddit dataset (Baumgartner et al. 2020; bulk dump as of March 2023). To create this initial dataset, each submission was extracted and concatenated it with its top-scoring, top-level comment. (In the case of tied top-scoring comments, we chose the longer of the two.) We then performed further rule-based filtering with the following constraints:
- Filter out deleted/removed content.
- Filter out content marked as over_18.
- Filter out all posts from a list of 26,123 banned or NSFW subreddits.
- Filter out posts from likely bot authors (drawn from https://botrank.pastimes.eu/ as of Sept 2024).
- Filter out posts containing non-text media.
- Perform document-level text deduplication via Bloom filter.

### 2. Retrieval-based subreddit selection

Dense retrieval was then used to identify academically-relevant subreddits for further filtering. We adapted search queries from MMLU test questions, and performed dense retrieval with these queries on the filtered Reddit data from Step #1, retaining the top 5 hits for each query. 

Based on these retrieved outputs, we then selected subreddits for inclusion in Step #3, filtering the dataset frm Step #1 to retain only documents from the selected subreddits. 

The specific criteria used to select the subreddits constitutes the primary difference between the high_relevance and low_relevance sets, and will be outlined below. 

### 3. Format rewriting 

Finally, the data from Step #2 was input to a synthetic rewriting pipeline to generate academic QA items with coverage of diverse question formats. 

We defined 7 categories of question structure inspired by variation observed in MMLU, and for each structure category we constructed a prompt template for generating questions of that category given an input text. Within the templates we then inserted texts drawn from the submission/comment data from Step #2. For each submission/comment text we sampled at least one format category to prompt for -- for longer input texts, structure categories were resampled and prompted for again, a number of times proportional to the length of the text. These prompts were submitted to GPT-4o mini.

### 4. Postprocessing

GPT-4o mini outputs were parsed into separate QA items based on the "%%%%" separator requested in the prompts. We kept all items containing the string "Answer: ".

---

## Distinguishing features

Below we list distinguishing features differing between reddit_to_flashcards-high_relevance and reddit_to_flashcards-low_relevance.

### high_relevance 

#### Subreddit selection criteria

For the high_relevance set, subreddits were selected if they met the following criteria:
- Subreddit has >= 20 unique retrieved items for queries within a given MMLU category; OR
- Subreddit has >=100 retrieved items for queries across all MMLU categories.
  
This yielded a list of 151 subreddits, which can be found in `subreddits-high-relevance.txt`.

#### Prompts

Prompt templates can be found in `prompts-high-relevance-set.py`.

#### Question structure sampling distribution

```
distribution = [
        (OPEN_ENDED,.17),
        (STATEMENT_COMPLETION,.17),
        (FILL_IN_BLANK,.17),
        (TWO_STATEMENT,.05),
        (WHICH_HAS_PROPERTY,.17),
        (WHICH_TRUE,.17),
        (IN_QUESTION_OPTIONS,.1)
    ]
```

#### Postprocessing

For this set, after generation was complete 50% of items were prepended with the prefix "Question: ".

### low_relevance

#### Subreddit selection criteria

For the low_relevance set, we included subreddits based on the following criterion:
- Subreddit has >= 5 retrieved items for for queries within a given MMLU category (in this case the criterion does not require *unique* items, such that the criterion can be met by the same item being retrieved 5 times for different queries within a category)

This yielded a set of 885 subreddits, and we took the 734 subreddits that were selected based on this threshold and not included in the 151 subreddits selected for the high_relevance set. The resulting list can be found in `subreddits-low-relevance.txt`.

#### Prompts

Prompt templates can be found in `prompts-low-relevance-set.py`.

The prompt templates for this set differ only in that the prefix "Question: " is included in the few-shot examples to encourage generation of items that use this prefix.

#### Question structure sampling distribution

```
distribution = [
        ('OPEN_ENDED',.25),
        ('STATEMENT_COMPLETION',.15),
        ('FILL_IN_BLANK',.15),
        ('TWO_STATEMENT',.05),
        ('WHICH_HAS_PROPERTY',.15),
        ('WHICH_TRUE',.15),
        ('IN_QUESTION_OPTIONS',.1)
    ]
```

#### Postprocessing

Since "Question: " was included in the few-shot examples for the prompts, this string was not prepended during postprocessing.

