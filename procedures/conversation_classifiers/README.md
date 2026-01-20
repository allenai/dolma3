# Classification of SFT data

Instructions on how to use the classifier

## Usage

This is a sample command

```shell
uv run \
  scripts/conversation_classifiers/conversation_classifiers.py \
  -d allenai/Wildchat-1M-gpt-4.1-regenerated-english-decontam-v2-filtered-tagged-topic \
  -o allenai/Wildchat-1M-gpt-4.1-regenerated-english-decontam-v2-filtered-tagged-topic-refusal \
  -t refusal
```

This runs the `refusal` tagger on `allenai/Wildchat-1M-gpt-4.1-regenerated-english-decontam-v2-filtered-tagged-topic` dataset, and puts the result in `allenai/Wildchat-1M-gpt-4.1-regenerated-english-decontam-v2-filtered-tagged-topic-refusal` (with extra column named "refusal" with new annotations).

**NOTE: you must run with `uv run` as shown above.**

## Customizing the tagger

You can change the model used with flag `--model <model_name>`. By default it uses GPT-5 mini.

You can switch classifiers with flag `-t <classifier_name>`. Options are:

- `topic`: tags according to 28 possible categories from [Chatterji, et al. (2025)][2]. **Default option**.
- `categories`: tags according some failure categories (incomplete requests, model self id, etc). **Not used**.
- `refusal`: tags messages if they are refusals or not, and, if yes, what kind of refusal they are. Uses taxonomy from [von Recum, et al. (2024)][1].
- `work`: tags requests as work or not work related. Uses taxonomy from [Chatterji, et al. (2025)][2].
- `ade`: partition requests based on whether they are asking the model to do something ("doing"),  ask for information ("asking"), or something else ("expressing"). Uses taxonomy from [Chatterji, et al. (2025)][2].


By default, script looks for a column named `messages` for a list of dicts (array of objects if u like javascript better) containing a conversation in OpenAI format (`dict[str, str]`, one key being `role` and the other `content`). You can use a different column with `-f <column_name>`.


If you want to run classifier on just a subset of the data, use `--limit N` to run on the first N rows.


## Tagging WildChat-4.8M

```bash
# set keys
export OPENAI_API_KEY=sk-...
export HF_XET_HIGH_PERFORMANCE=1

# caching
uv run --with=huggingface-hub hf download allenai/WildChat-4.8M --repo-type dataset

# do the actual tagging
uv run scripts/topic_sft/openai_classifier.py \
    --dataset-dir allenai/WildChat-4.8M \
    --output-dir soldni/WildChat-4.8M-topic \
    --batch-size 5000 \
    --service-tier flex \
    --model-name gpt-5-mini \
    --conversation-field conversation \
    --task-prompt topic
```

## Tagging LMSYS-1M

```bash
# set keys
export OPENAI_API_KEY=sk-...
export HF_XET_HIGH_PERFORMANCE=1

# caching
uv run --with=huggingface-hub hf download lmsys/lmsys-chat-1m --repo-type dataset

# do the actual tagging
uv run scripts/topic_sft/openai_classifier.py \
    --dataset-dir lmsys/lmsys-chat-1m \
    --output-dir soldni/lmsys-chat-1m-topic \
    --batch-size 5000 \
    --service-tier flex \
    --model-name gpt-5-mini \
    --conversation-field conversation \
    --task-prompt topic
```

---
[1]: https://arxiv.org/abs/2412.16974
[2]: https://www.nber.org/papers/w34255
