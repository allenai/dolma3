# 9T Pretraining Pool

## Table of Contents

- [Common Crawl](#common-crawl)
- [olmOCR Science PDFs](#olmocr-science-pdfs)
- [StackEdu](#stackedu)
- [arXiv](#arxiv)
- [FineMath 3+](#finemath-3)
- [Wikipedia & Wikibooks](#wikipedia--wikibooks)

---

Here we discuss each of the sources and steps taken to curate the 9T pretraining pool. We summarize each of these sources and their sizes in the following table:

| Source | Type | 9T Pool Tokens | 9T Pool Docs |
|--------|------|----------------|--------------|
| Common Crawl | Web pages | 8.14T | 9.67B |
| olmOCR Science PDFs | Academic documents | 972B | 101M |
| StackEdu (Rebalanced) | GitHub code | 137B | 167M |
| arXiv | Papers with LaTeX | 21.4B | 3.95M |
| FineMath 3+ | Math web pages | 34.1B | 21.4M |
| Wikipedia & Wikibooks | Encyclopedic | 3.69B | 6.67M |
| **Total** | | **9.31T** | **9.97B** |

---

## Common Crawl

There are many steps to processing Common Crawl to be amenable for pretraining, which we will cover in detail. To summarize these steps, we outline the recipe as follows:

* **Linearization:** Common Crawl provides data in terms of raw web archive data, riddled with HTML tags and other metadata not directly helpful for training. Linearization is the process of text extraction from these WARC/WET files.
* **Heuristic Filtering:** To convert the large, linearized dataset into something much more manageable, we apply a round of highly-efficient, but coarse, filtration techniques to remove data that is of low-quality or non-English.
* **Deduplication:** Duplicate data has been shown to be an inefficient use of training tokens. We apply three rounds of deduplication: first an exact-deduplication to remove identical documents; then a MinHash-based fuzzy deduplication to remove documents that are "nearly identical"; then a suffix-array based deduplication to remove repeated substrings within the surviving documents.
* **Categorization and Quality Partitioning:** We apply two separate classifiers to each remaining document to first classify the data into [WebOrganizer categories](https://arxiv.org/abs/2502.10341) and into quality buckets within each category. This coarse-grained partitioning of the data makes it amenable to mixing and upsampling.
* **Mixing:** The natural data distribution of Common Crawl isn't optimized for language model pretraining. To rectify this, we filter and upsample documents according to their category and quality bucket from the previous step.

### Linearization

We start with 104 Common Crawl dumps, ranging from 2013 to 2024, with a cutoff date of December 31, 2024. The full list of Common Crawl dumps used is:

```
['CC-MAIN-2013-20', 'CC-MAIN-2013-48', 'CC-MAIN-2014-10', 'CC-MAIN-2014-15', 
'CC-MAIN-2014-23', 'CC-MAIN-2014-35', 'CC-MAIN-2014-41', 'CC-MAIN-2014-42', 
'CC-MAIN-2014-49', 'CC-MAIN-2014-52', 'CC-MAIN-2015-06', 'CC-MAIN-2015-11', 
'CC-MAIN-2015-14', 'CC-MAIN-2015-18', 'CC-MAIN-2015-22', 'CC-MAIN-2015-27', 
'CC-MAIN-2015-32', 'CC-MAIN-2015-35', 'CC-MAIN-2015-40', 'CC-MAIN-2015-48', 
'CC-MAIN-2016-07', 'CC-MAIN-2016-18', 'CC-MAIN-2016-22', 'CC-MAIN-2016-26', 
'CC-MAIN-2016-30', 'CC-MAIN-2016-36', 'CC-MAIN-2016-40', 'CC-MAIN-2016-44', 
'CC-MAIN-2016-50', 'CC-MAIN-2017-04', 'CC-MAIN-2017-09', 'CC-MAIN-2017-13', 
'CC-MAIN-2017-17', 'CC-MAIN-2017-22', 'CC-MAIN-2017-26', 'CC-MAIN-2017-30', 
'CC-MAIN-2017-34', 'CC-MAIN-2017-39', 'CC-MAIN-2017-43', 'CC-MAIN-2017-47', 
'CC-MAIN-2017-51', 'CC-MAIN-2018-05', 'CC-MAIN-2018-09', 'CC-MAIN-2018-13', 
'CC-MAIN-2018-17', 'CC-MAIN-2018-22', 'CC-MAIN-2018-26', 'CC-MAIN-2018-30', 
'CC-MAIN-2018-34', 'CC-MAIN-2018-39', 'CC-MAIN-2018-43', 'CC-MAIN-2018-47', 
'CC-MAIN-2018-51', 'CC-MAIN-2019-04', 'CC-MAIN-2019-09', 'CC-MAIN-2019-13', 
'CC-MAIN-2019-18', 'CC-MAIN-2019-22', 'CC-MAIN-2019-26', 'CC-MAIN-2019-30', 
'CC-MAIN-2019-35', 'CC-MAIN-2019-39', 'CC-MAIN-2019-43', 'CC-MAIN-2019-47', 
'CC-MAIN-2019-51', 'CC-MAIN-2020-05', 'CC-MAIN-2020-10', 'CC-MAIN-2020-16', 
'CC-MAIN-2020-24', 'CC-MAIN-2020-29', 'CC-MAIN-2020-34', 'CC-MAIN-2020-40', 
'CC-MAIN-2020-45', 'CC-MAIN-2020-50', 'CC-MAIN-2021-04', 'CC-MAIN-2021-10', 
'CC-MAIN-2021-17', 'CC-MAIN-2021-21', 'CC-MAIN-2021-25', 'CC-MAIN-2021-31', 
'CC-MAIN-2021-39', 'CC-MAIN-2021-43', 'CC-MAIN-2021-49', 'CC-MAIN-2022-05', 
'CC-MAIN-2022-21', 'CC-MAIN-2022-27', 'CC-MAIN-2022-33', 'CC-MAIN-2022-40', 
'CC-MAIN-2022-49', 'CC-MAIN-2023-06', 'CC-MAIN-2023-14', 'CC-MAIN-2023-23', 
'CC-MAIN-2023-40', 'CC-MAIN-2023-50', 'CC-MAIN-2024-10', 'CC-MAIN-2024-18', 
'CC-MAIN-2024-22', 'CC-MAIN-2024-26', 'CC-MAIN-2024-30', 'CC-MAIN-2024-33', 
'CC-MAIN-2024-38', 'CC-MAIN-2024-42', 'CC-MAIN-2024-46', 'CC-MAIN-2024-51']
```

Then we apply [Resiliparse](https://resiliparse.chatnoir.eu/en/stable/) extraction to remove HTML artifacts. For most of these crawls, we use the publicly available Resiliparse-extracted dumps from the [DCLM Pool](https://data.commoncrawl.org/contrib/datacomp/DCLM-pool/index.html). For the remaining dumps, we use the [Dolma toolkit](https://github.com/allenai/dolma) to extract the text from the WARC files. After this step, we have 255.7B documents in total.

### Heuristic Filtering

Even after HTML artifacts are removed, datasets are often too large and contain mostly low-quality data that is easily identifiable. Following suit of several other common processing pipelines (see RefinedWeb, DCLM, FineWeb), we apply a set of heuristic filters to catch and remove much of this obviously bad data, in addition to performing language filtering to select only English documents. We use a [bespoke native-Rust toolkit](https://github.com/allenai/datamap-rs) to handle these steps. The sub-steps that occur during the heuristic filtering pipeline can be broken down into components like:

* **URL Filtration:** Filter documents based on the contents of their URLs based on various blacklists. This step mostly removes web pages that obviously contain spam or adult content. This removes ~2% of the pool.
* **Basic Heuristics:** This step removes documents that are obviously low-quality. This includes web pages that are too short, too long, don't contain enough alphanumeric characters, or contain too many stopwords. The majority of the pool does not survive this step, with ~64% of the total pool being filtered after this step.
* **Repetition Filter:** We remove documents that contain large amounts of internally repetitious text. This follows the same procedures as in the [Gopher paper](https://arxiv.org/pdf/2112.11446). This step removes ~12% of the total pool.
* **English Filter:** We then apply an English-language filter to the remaining documents. We use the [`lid.176.bin`](https://fasttext.cc/docs/en/language-identification.html) FastText model to identify the language of each document and keep only documents that have an English score of at least 0.65. This step removes ~3% of the total pool. We note that this is a much smaller removal than the English filtration steps of RefinedWeb or DCLM—we conjecture this is because we applied cheaper heuristics first: English filtering is comparatively expensive and filtration steps are commutative, so it makes sense to perform the cheaper steps first.
* **MadLad Filtration:** We apply one last round of filtration, based on the rules for identifying "questionable" sentences from the [Madlad400 paper](https://arxiv.org/pdf/2309.04662). Ablations demonstrated that it was most effective to only apply rules 2 and 5 (identifying sentences with weird up/downcasing and cursed regexes) and remove documents with at least 20% questionable sentences. This removed another ~3% of the total pool.
* **Modifications:** Finally we applied a round of text modifiers that do not filter the data but do remove or modify substrings within the corpus, such as many repeated newlines or commonly repeated filler words like "Read more..." or "items in cart". This does not reduce the number of documents in the corpus, but does remove some obvious low-utility text.

After fully applying this pipeline to each of our 104 Common Crawl shards, we have a corpus of 38.7B documents, an 85% reduction in the input corpus size.

### Deduplication

We perform deduplication in three rounds of varying coarseness. The first round identifies and removes any documents that are exact copies of each other, down to byte-wise equality over their content strings only. The heuristic filtering step annotates each jsonl with hashes according to the text content, and these hashes are used to perform exact deduplication. We note that exact deduplication can be subsumed entirely by a MinHash fuzzy deduplication step. However, we found that the quantity of exact-duplicates in Common Crawl is large enough that it is worth it, in terms of compute efficiency, to perform exact-deduplication to prune the pool before running the more expensive MinHash fuzzy deduplication.

#### Exact Deduplication

During the heuristic filtering phase, we imbued each document with a hash of its text content. We leverage this to remove duplicate content from the dataset by first grouping documents according to this hash value, such that all documents within a given hash value live in a known set of files. Then we load each of these groups into memory and keep only one document per unique hash value. This can be easily done with the [Duplodocus tool](https://github.com/allenai/duplodocus), with either the `exact-dedup-memory` or `exact-dedup-disk-...` commands.

This exact deduplication phase removes 67% of the remaining pool, yielding a corpus of 12.8B documents.

#### MinHash Deduplication

Next we apply a MinHash deduplication procedure to remove documents that are not exact-duplicates, but are nearly exact-duplicates. Examples of the types of documents that get targeted during a MinHash deduplication phase are documents that contain the same body text but differ in only a few words in the header or footer of the document. We partition the dataset into 32 shards of roughly equal size and perform the following procedure on each step independently. We perform MinHash using an ngram size of 5, and 26 buckets of size 11 for a total of 286 hashes per ngram. This identifies documents with a Jaccard similarity of 0.8 with probability 90%, and a pair of documents with a Jaccard similarity of 0.6 have a 10% chance of being marked as duplicates.

After identifying and annotating the duplicate cluster of each document, we proceed to check pairs of documents in each cluster for their true Jaccard similarity, filtering for Jaccard similarity of 0.8. Amongst clusters of documents that truly have a high Jaccard similarity, we keep only the most recent document, according to crawl date. This procedure removes a total of 23% of the remaining pool of data.

This step can be performed by using the `minhash-memory` or `mh-...` commands within the [Duplodocus tool](https://github.com/allenai/duplodocus).

#### Substring Deduplication

Finally we run a substring deduplication procedure to identify and remove repeated substrings that occur within numerous documents in our corpus. This step is intended to remove repetitious boilerplate text, such as repeated headers and footers within web pages. This is performed by sharding the dataset into 56 roughly equal shards and building a suffix array over each shard. Any substring of length at least 500 bytes is then identified and marked for deletion. We apply a novel fuzzy deduplication step where we also remove any short substrings that lie in-between repeated intervals. We also apply the novel step where we ensure we keep at least one copy of each substring within the corpus. Ultimately this step removes 14% of the remaining text.

This step can be performed by using the [bsade tool](https://github.com/liujch1998/bsade/) to identify repeated substrings, and the [datamap-rs tool](https://github.com/allenai/datamap-rs) to perform the fuzzy filtering step.

After all deduplication steps have been applied, we are left with a corpus of 9.7B documents.

### Categorization and Quality Partitioning

With an aggressively deduplicated corpus of web text, we apply quality and topic classifiers to partition the dataset into buckets of `(topic, quality)` to be used in a fine-grained mixing procedure. To classify documents according to topic, we use the topics defined by [WebOrganizer](https://arxiv.org/abs/2502.10341). Classification is performed using a FastText classifier. To imbue all documents with a quality score, we apply a custom quality classifier, not too dissimilar from that used by DCLM. To further partition each topic bucket into quality buckets, we run a reservoir sample over the quality scores within each bucket and define ranges roughly corresponding to 5-percentile intervals ("vigintiles"). All steps here can be performed by repeated applications of commands within the [datamap-rs tool](https://github.com/allenai/datamap-rs).

---

## olmOCR Science PDFs

This document describes the complete pipeline for processing PDF documents for pretraining, from initial collection through final quality filtering. The pipeline transforms raw PDF files into clean, deduplicated, and categorized text suitable for language model training.

### Pipeline Overview

The PDF processing pipeline consists of the following major steps:

* **Dataset Origin & Collection:** Starting from a large corpus of PDF documents crawled from scientific and research-oriented URLs via the Semantic Scholar Anansi pipeline.
* **OCR & Text Extraction:** Converting PDF files to text using olmOCR with fallback mechanisms for robust processing.
* **Deduplication:** Applying MinHash-based fuzzy deduplication to remove near-duplicate documents.
* **PII Filtering:** Using specialized language models to identify and remove documents containing personally identifiable information.
* **Quality Filtering:** Applying heuristic filters to remove low-quality documents based on language detection, content characteristics, and structural properties.
* **Domain Classification:** Categorizing documents using WebOrganizer to enable fine-grained mixing.
* **Legal Filtering:** Removing documents based on legal deny lists.
* **Final Quality Enhancement:** Additional filtering based on compression ratios and other metrics for mid-training and long-context scenarios.

### Dataset Origin

#### Initial Collection

The dataset originates from **238,475,382 unique PDF hashes (SHA-1)** crawled from various scientific and research-heavy, publicly available URLs by the **Semantic Scholar Anansi pipeline**. The crawler has been acquiring PDFs for over 10 years following good crawler practices:

- Disclosed agent
- No paywall circumventing
- Respectful crawling rates

#### Source Query

The initial PDF hashes were collected using an Athena query that:

1. Filters PDFs to only those exclusively from specific trusted sources (Anansi, Unpaywall, various extraction pipelines, etc.)
2. Identifies the earliest URI from `anansi.pdf_fetch` with fetch dates
3. Falls back to URI and accessed date from `content_ext.sourced_papers.obj` when primary source unavailable
4. Combines and prioritizes the most appropriate date for each PDF

The full SQL query can be found in the source documentation and filters documents to ensure they come from legitimate, non-paywalled sources.

### OCR & Text Extraction

#### olmOCR Processing

PDFs were processed into Dolma document format using **olmOCR** (versions 0.1.49 to 0.1.53), an early version of the pipeline before public release. The process used:

- **Model:** `allenai/olmOCR-7B-0225-preview`
- **Pipeline:** Early version with similar weights but different code from the public release

See: [olmOCR paper](https://arxiv.org/abs/2502.18443) and [model on HuggingFace](https://huggingface.co/allenai/olmOCR-7B-0225-preview)

#### Initial Filtering During Extraction

During the olmOCR processing stage, basic spam and language filtering was applied:

- **Language Detection:** Using the Lingua language detector
- **Spam Detection:** Checking for frequency of common spam keywords
- Documents failing these filters were not converted with olmOCR

Filter code: [olmocr/filter/filter.py](https://github.com/allenai/olmocr/blob/v0.1.53/olmocr/filter/filter.py)

#### Fallback Mechanism

olmOCR included a robust fallback mechanism:

- Allowed up to **1 out of every 250 pages** to fail processing (e.g., JSON parsing errors)
- Failed pages had content extracted using poppler's `pdftotext` tool
- Documents with more than 1 error page per 250 total pages were skipped entirely

#### Output

**Total documents after extraction: 160,027,785**

This represents approximately 33% removed from the original 238M documents due to:

- Language filtering (~25%)
- Spam filtering (~5%)
- Processing failures (~3%)

### Deduplication

A MinHash-based deduplication was applied to remove near-duplicate documents using the [Duplodocus tool](https://github.com/allenai/duplodocus). The same parameters were used as in FineWeb's deduplication procedure: ngrams of size 5 using the p50k tokenizer were used, with 14 bands of size 5. After deduplication, source URLs were added back by matching against the original Anansi crawler queries.

**Total documents after deduplication: 156,410,102** (2.2% reduction)

### PII Filtering

PII (Personally Identifiable Information) filtering was one of the most complex and carefully designed steps in the pipeline.

#### Initial Approach: Human Annotation (Failed)

The team initially attempted to hire annotators via Prolific with detailed instructions to identify documents containing PII. However, this approach failed due to:

- Annotators rushing through documents without careful review
- Random over-annotation and under-annotation
- Difficulty explaining contextual nuances (e.g., public records vs. private information)

#### Evolution to Model-Based Approach

The team evolved through several approaches:

1. **GPT-4.1:** Better understanding of subtleties and consistent performance, but expensive
2. **Distillation:** Rules distilled to simpler prompts for smaller, local models
3. **Two-stage classification:** Combining document type classification with detailed PII detection

#### Stage 1: Document Type Classification

Using **google/gemma-3-4b-it** to classify the first 5,000 characters of each document:

**Prompt:**
```
Given the text above, determine what type of document it is. Answer in JSON.
The format of your json object should be {'primary_language': str,
'document_type': str, 'is_resume_cv': bool, 'is_academic_paper': bool,
'is_textbook': bool, 'is_news_article': bool, 'is_test_or_quiz': bool,
'is_homework_assignment': bool, 'is_class_syllabus': bool,
'is_meeting_minutes': bool, 'is_legal_contract': bool, 'is_form': bool,
'is_correspondence_or_letter': bool, 'is_public_order': bool,
'is_court_notice': bool, 'is_completion_certificate': bool, 'contains_pii':
bool}
```

**Performance:** 17,000 input tokens/sec per H100, 1,000 output tokens/sec

#### Stage 2: Detailed PII Classification

Using **google/gemma-3-12b-it** with a more detailed prompt analyzing the first page:

**Key Guidelines:**

**PII That Occurs Even Without an Identifier:**

- Government IDs (SSN, passport numbers, driver's license numbers, tax IDs)
- Financial information (credit card numbers, bank account/routing numbers)
- Biometric data (fingerprints, retina scans, facial recognition data, voice signatures)
- Login information (ONLY when username, password, and login location present together)

**Identifiers for PII:**

- Names (full names, first names, last names, nicknames)
- Email addresses
- Phone numbers

**PII That Must Co-Occur With an Identifier:**

- Addresses
- Biographical information (DOB, place of birth, gender, sexual orientation, race, ethnicity, citizenship, religion)
- Location information (geolocations, specific coordinates)
- Employment information (job titles, workplace names, employment history)
- Education information (school names, degrees, transcripts)
- Medical information (health records, diagnoses, genetic or neural data)

**Special Rules:**

- For forms, ONLY filled-out fields count as PII
- References to documents that typically contain PII don't themselves contain PII
- Only actual occurrences within the document count

**Performance:** 12,000 input tokens/sec per H100, 2,000 output tokens/sec

Both models used VLLM with structured decoding for efficient processing.

**Scripts:**

- [pii_rule_comparison.py](https://github.com/allenai/olmocr/blob/v0.1.76/scripts/pii_rule_comparison.py)
- [rich_tagging_pipeline.py](https://github.com/allenai/olmocr/blob/v0.1.76/scripts/rich_tagging_pipeline.py)
- [tagging_pipeline_v2.py](https://github.com/allenai/olmocr/blob/v0.1.76/scripts/tagging_pipeline_v2.py)

#### Final PII Filter Rules

Applied using Dolma toolkit with jq syntax:

**Filter Configuration:** [pii_removal.yml](https://github.com/allenai/dolma/blob/jakep-s2pdfs/configs/s2pdfs/pii_removal.yml)

**Logic:**

Remove if:
```
(
  (contains_pii == true) AND
  (is_public_document != true) AND
  (NOT academic_paper) AND
  (NOT textbook) AND
  (NOT homework_assignment) AND
  (NOT test_or_quiz) AND
  (NOT class_syllabus) AND
  (NOT public_order) AND
  (NOT news_article)
) OR
(is_resume_cv == true) OR
(is_court_notice == true) OR
(is_completion_certificate == true)
```

**Total documents after PII filtering: 148,665,265** (4.9% reduction)

### Basic Quality Filtering

After PII removal, manual review revealed large sections of low-quality documents including:

- Non-English documents that passed the initial filter
- Documents with excessive tables
- Documents with too many numbers compared to English text

#### Quality Filter Configuration

Applied using Dolma toolkit: [basic_quality.yml](https://github.com/allenai/dolma/blob/jakep-s2pdfs/configs/s2pdfs/basic_quality.yml)

**Attributes Computed:**

- `avg_fraction_numbers_in_line_v1`: Average fraction of numeric characters per line
- `fineweb_edu_fasttext_gt2`: Educational quality score from FineWeb
- `ft_lang_id_en_doc_v2`: English language identification score
- `pipe_delimited_lines_v1`: Ratio of lines with pipe delimiters (often tables)

**Filter Rules:**

Keep documents where:
```
(ft_lang_id_en_doc_v2 > 0.5) AND
(fineweb_edu_fasttext_gt2 > 0.001) AND
(avg_fraction_numbers_in_line < 0.2) AND
(pipe_delimited_lines_ratio < 0.3)
```

**Output:**
```
s3://ai2-llm/pretraining-data/sources/s2pdf_dedupe_minhash_v1_with_no_pii_basic_quality/
```

**Total documents after quality filtering: 108,354,688** (27.1% reduction)  
**Documents with null/missing Source-URL: 4,497,442**

### Domain Classification with WebOrganizer

To enable fine-grained mixing strategies, documents were categorized using the [WebOrganizer](https://arxiv.org/abs/2502.10341) taxonomy.

**Classification Method:** FastText classifier trained on WebOrganizer categories

See: [WebOrganizer project](https://weborganizer.allen.ai/)

### Legal Filtering & Deny Lists

#### First Legal Filter (OLMo3-7B)

Documents were filtered based on legal deny lists to remove content that should not be included for legal/copyright reasons.

**Total documents: 107,987,567** (0.3% reduction)  
**Documents with null/missing Source-URL: 4,482,249**

**→ Used for OLMo3-7B training**

#### Additional Legal Filter (OLMo3-32B)

An additional legal filtering pass was performed for the OLMo3-32B training.

**Total documents: 100,875,901** (6.6% reduction)  
**Documents with null/missing Source-URL: 0**

**→ Used for OLMo3-32B training**

### Pipeline Summary Statistics

| Step | Total Documents | Documents with No Source-URL | Reduction |
|------|-----------------|------------------------------|-----------|
| Initial Athena Query from S2 | 238,475,382 | - | - |
| OCR + Language/Spam Filter | 160,027,785 | - | 32.9% |
| MinHash Deduplication | 156,410,102 | - | 2.2% |
| PII Filtering | 148,665,265 | 5,165,558 | 4.9% |
| Basic Quality Filtering | 108,354,688 | 4,497,442 | 27.1% |
| Legal Deny List Filtering | 107,987,567 | 4,482,249 | 0.3% |
| **→ Used for OLMo3-7B** | **107,987,567** | **4,482,249** | - |
| Additional Legal Filter | 100,875,901 | 0 | 6.6% |
| **→ Used for OLMo3-32B** | **100,875,901** | **0** | - |

**Overall Reduction:** 57.7% from initial collection to OLMo3-32B training set

---

## StackEdu

For code training data, we used [Stack-Edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu) directly, but we rebalanced the programming language mix during the mixing phase.

---

## arXiv

For extra mathematical documents, we include arXiv documents from the [Proof-Pile-2 dataset](https://huggingface.co/datasets/EleutherAI/proof-pile-2).

---

## FineMath 3+

For even more mathematical content, we use [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath), but only include documents that have a score from the FineMath classifier of at least 3.

---

## Wikipedia & Wikibooks

For encyclopedic content, we include the Wikipedia and Wikibooks sources from [Dolma 1.7](https://huggingface.co/datasets/allenai/dolma).