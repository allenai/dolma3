## Ingestion

Our ingested web text consists of the warc files of 104 crawls from CC-MAIN-2013-20 to CC-MAIN-2024-51 accessible at s3://commoncrawl/crawl-data.

In its raw form, this dataset is 6.2 PB. However, roughly 94% of this text is HTML artifacts, so we relied on [resiliparse](https://resiliparse.chatnoir.eu/en/latest/man/extract/html2text.html#basic-plain-text-conversion) to extract the plain text with semantic value present in each page.

For crawls through CC-MAIN-2022-49, we copied the resiliparse-linearized data available at s3://commoncrawl/contrib/datacomp/DCLM-pool/jsonl/ provided as part of the [DataComp project](https://data.commoncrawl.org/contrib/datacomp/DCLM-pool/index.html) . 

We linearized the remaining crawls using [Dolma](https://github.com/allenai/dolma/blob/45482814db21e79df9fa7b6ee7f1270839976472/python/dolma/warc/processor.py#L224), distributed across 50 ECS tasks each with 2 vCPUs and 4 GB RAM running over the course of 6 days. 

This extracted 350 TB of linearized text.  

