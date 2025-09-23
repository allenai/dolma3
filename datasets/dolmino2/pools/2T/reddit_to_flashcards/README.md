

## Common processing steps

Both reddit_to_flashcards-high_relevance and reddit_to_flashcards-low_relevance have the following properties.

### 1. Preliminary filtering

A single dataset of submission/comment pairs was derived from the PushShift Reddit dataset (Baumgartner et al. 2020; bulk dump as of March 2023). Each submission was extracted and concatenated it with its top-scoring, top-level comment. (In the case of tied top-scoring comments, we chose the longer of the two.) We then performed further rule-based filtering with the following constraints:
- Filter out deleted/removed content.
- Filter out content marked as over_18.
- Filter out all posts from a list of 26,123 banned or NSFW subreddits.
- Filter out posts from likely bot authors (drawn from https://botrank.pastimes.eu/ as of Sept 2024).
- Filter out posts containing non-text media.
- Perform document-level text deduplication via Bloom filter.

### 2. Retrieval-based subreddit selection

Dense retrieval was then used to identify academically-relevant subreddits for further filtering. We adapted search queries from MMLU test questions, and performed dense retrieval with these queries on the filtered Reddit data from Step #2, retaining the top 5 hits for each query. 

Based on these retrieved outputs, we then selected subreddits for inclusion in the dataset. The criteria used to select subreddits is the prinary difference between the high_relevance and low_relevance sets, as described below. 

We then filtered the dataset from Step #1 to retain only documents from subreddits on this list of 151 subreddits.

### 3. Format rewriting 

Finally, the data from Step #2 was input to a synthetic rewriting pipeline to generate academic QA items with coverage of diverse question formats. 

We defined 7 categories of question format inspired by variation observed in MMLU, and used these to construct prompts for QA text generation. The format categories are as follows:
1. open-ended
2. statement completion
3. fill-in-the-blank
4. statement truth verification
5. which-of-following-has-property-X
6. which-of-following-is-true
7. in-question options

For each format category we constructed a prompt for generating questions of that category given an input text. Below is an example prompt, for the "in-question-options" category. Prompts for other categories differ in 1) the content of the "For format ..." paragraph and 2) the in-context examples (1-3 examples per prompt).

For generating our rewritten QA data, we prompted GPT-4o mini (Jan 2025 version). We iterated over the submission/comment pairs in the data from Step #2, and for each of these texts we sampled a format category and prompted the GPT-4o mini to generate QA pairs for that text and format category. For longer input texts, format categories were resampled and prompted for again, a number of times proportional to the length of the text. 

Finally, GPT-4o mini outputs were parsed into separate QA items based on the "%%%%" separator, keeping all items containing "Answer: ".


---

## Distinguishing features

Below we list distinguishing features in reddit_to_flashcards-high_relevance and reddit_to_flashcards-low_relevance.

### high_relevance

Subreddit has >= 20 unique retrieved items for queries within a given MMLU category; OR
Subreddit has >=100 retrieved items for queries across all MMLU categories.
We then filtered the dataset from Step #1 to retain only documents from subreddits on this list of 151 subreddits.

distribution = [
        (OPEN_ENDED,.17),
        (STATEMENT_COMPLETION,.17),
        (FILL_IN_BLANK,.17),
        (TWO_STATEMENT,.05),
        (WHICH_HAS_PROPERTY,.17),
        (WHICH_TRUE,.17),
        (IN_QUESTION_OPTIONS,.1)
    ]

50% of items were prepended with the prefix "Question: ".

### low_relevance

Low threshold selects 885 subreddits based on whether subreddit has >= 5 retrieved items for any MMLU category -- this one did NOT deduplicate so it could be the same item 5 times for different queries

We took the 734 subreddits that were selected based on this threshold and not included in the 151 subreddits selected by the high threshold


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




