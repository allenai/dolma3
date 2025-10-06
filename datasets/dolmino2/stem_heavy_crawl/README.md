#STEM Heavy Crawl 

##Data Discovery
The 'STEM Heavy Crawl' data was crawled between 9/12/2024-6/3/2025 by the Ai2 pipeline. The crawler followed good crawler practices (disclosed agent, no paywall circumventing, etc) and made requests with the Ai2 User-Agent header:

	Mozilla/5.0 (compatible) AI2Bot (+https://www.allenai.org/crawler) 

The crawler ingested various scientific, educational, and general domains based on domain-level 'seeds' sourced from manual lists of websites deemed high value (see seed_sets.txt). From seeded domains, pages were prioritized for crawling both manually and based on the presence of pages with a higher code prose score. 


##'Code Prose' Evaluation

A code prose [FastText](https://fasttext.cc/docs/en/supervised-tutorial.html) classifier  was used to identify document spans as code, prose, or other. Based on these evaluations, we determined if a document met two sets of conditions:

Code Prose:

    code_composition > 0.05
    prose_composition > 0.3
    code_count >= 10
    code_mean_entropy < 0.4

Code:
	
	code_composition > 0.5 
	code_count >= 20
	code_mean_entropy < 0.4

A domain's crawl priority was increased in proportion to the amount of 'code' and 'code prose' documents crawled so far from that domain. 


##Data Processing

Plain text data crawled from the seeded domains was stored as WARC files and then linearized using [Resiliparse](https://resiliparse.chatnoir.eu/en/stable/man/extract/html2text.html#basic-plain-text-conversion) via [Dolma](https://github.com/allenai/dolma/blob/45482814db21e79df9fa7b6ee7f1270839976472/python/dolma/warc/processor.py). 

##Filtering


The linearized data was filtered for spam, length and forbidden domains using [datamap-rs](https://github.com/allenai/datamap-rs) with the [all-dressed config_v3](https://github.com/allenai/datamap-rs/blob/5ae3208a38247167112e9063a245673febb1a5f1/examples/all_dressed/config_v3.yaml) configuration. 

The dataset was then further filtered down to documents with a .6 [dclm\_oh\_eli5](https://huggingface.co/mlfoundations/fasttext-oh-eli5) score or greater.
