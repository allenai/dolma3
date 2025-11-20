#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "datasets>=3,<4",
#     "tqdm",
#     "click",
#     "smart_open[s3]",
#     "msgspec",
#     "tokenizers<=0.19.1",
#     "huggingface-hub[hf_transfer]",
#     "cached-path",
#     "blingfire",
#     "pydantic",
#     "openai>=1.0.0",
#     "boto3",
#     "pyonmttok",
#     "regex",
#     "boto3-stubs[s3]>=1.37.0.0"[],
#     "scikit-learn",
#     "blingfire",
#     "tiktoken",
#     "number-parser",
# ]
# ///
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import cached_property
import re
import math
import collections
from contextlib import ExitStack
import string
from urllib.parse import urlparse
import dataclasses
import regex
import hashlib
import json
import datetime
import random
import numpy as np
import fnmatch
import sys
from pathlib import Path
from concurrent.futures import as_completed
from typing import (
    NamedTuple,
    Protocol,
    Generator,
    ClassVar,
    TYPE_CHECKING,
    Optional,
    TypedDict,
    Generic,
    TypeVar,
    Literal,
)
from typing import TextIO
import warnings
import click
import tqdm

import smart_open
import tiktoken
import msgspec
import pyonmttok
from pydantic import BaseModel as PydanticBaseModel
from pydantic_core import ValidationError as PydanticValidationError
import boto3
import number_parser

from sklearn.feature_extraction.text import TfidfVectorizer


if TYPE_CHECKING:
    from openai import OpenAI  # noqa: F401
    from openai.types.batch import Batch  # noqa: F401
    from mypy_boto3_s3 import Client as S3Client


warnings.filterwarnings(
    "ignore",
    message=".*stop_words may be inconsistent.*",
    category=UserWarning,
    module="sklearn.feature_extraction.text",
)

warnings.filterwarnings(
    "ignore",
    message="The parameter 'token_pattern' will not be used since 'tokenizer' is not None'",
    category=UserWarning,
    module="sklearn.feature_extraction.text",
)


class BaseModel(PydanticBaseModel):
    class Config:
        extra = "forbid"


class DatasetRegistry:
    _registry: dict[str, str] = {}

    @classmethod
    def add(cls, name: str, path: str):
        cls._registry[name] = path

    @classmethod
    def get(cls, name: str) -> list[tuple[str, str]]:
        if re.escape(name) != name:
            # using regex matching
            return [(k, v) for k, v in cls._registry.items() if re.search(name, k)]
        else:
            return [(name, cls._registry[name])] if name in cls._registry else []


def get_nouns() -> set[str]:
    """
    from https://github.com/david47k/top-english-wordlists/blob/master/top_english_nouns_lower_10000.txt
    """

    filepath = Path(sys.path[0]) / "data/more_nouns10k.txt"
    with open(filepath, "r", encoding="utf-8") as f:
        nouns = [line.strip().lower() for line in f.readlines()]

    return set(nouns)


def get_stopwords() -> set[str]:
    """
    from https://github.com/igorbrigadir/stopwords/blob/master/en/terrier.txt
    """

    filepath = Path(sys.path[0]) / "data/terrier_stopwords.txt"
    with open(filepath, "r", encoding="utf-8") as f:
        stopwords = [line.strip().lower() for line in f.readlines()]

    return set(stopwords)


def format_to_dolma_timestamp(timestamp: datetime.datetime | None = None) -> str:
    """Format a timestamp as a string using near ISO-8601 format."""
    if timestamp is None:
        timestamp = datetime.datetime.now()
    return timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"


NOUNS = get_nouns()
STOPWORDS = get_stopwords()
# add_gpt5()


def base_model_to_json(model: BaseModel, indent: int = 2) -> str:
    return model.model_dump_json(indent=indent).replace("{", "{{").replace("}", "}}")


def base_model_to_dict(model: PydanticBaseModel) -> dict:
    return json.loads(model.model_dump_json(indent=2))


class CweQuestion(BaseModel):
    word: str
    count: int
    question: str
    answer: str


class CweOutputFormat(BaseModel):
    review_type: str
    review_questions: list[CweQuestion]


class CweCommonNouns(BaseModel):
    word: str
    count: int


class CweInputFormat(BaseModel):
    common_nouns: list[CweCommonNouns]


example1_passage = """
We are proud to open a new wing in the museum of fine arts. The wing contains numerous art pieces dating back to the 18th century. Visitors will discover masterful paintings, intricate sculptures, and rare manuscripts that showcase the artistic brilliance of European masters. This remarkable collection represents years of dedicated curation efforts and generous donations...
"""

example1_input = CweInputFormat(
    common_nouns=[
        CweCommonNouns(word="painting", count=50),
        CweCommonNouns(word="sculpture", count=47),
        CweCommonNouns(word="oil", count=30),
        CweCommonNouns(word="marble", count=26),
        CweCommonNouns(word="wing", count=19),
        CweCommonNouns(word="museum", count=17),
        CweCommonNouns(word="gallery", count=14),
        CweCommonNouns(word="rembrandt", count=13),
        CweCommonNouns(word="art", count=13),
        CweCommonNouns(word="exhibition", count=12),
    ]
)

example1_output = CweOutputFormat(
    review_type="Indexing of items in the museum.",
    review_questions=[
        CweQuestion(
            word="painting",
            count=50,
            question="What is the number of paintings in the museum?",
            answer="50",
        ),
        CweQuestion(
            word="sculpture",
            count=47,
            question="How many sculptures are in the museum?",
            answer="I've counted 47 of them!",
        ),
        CweQuestion(
            word="marble",
            count=26,
            question="Marble is used for how many art pieces?",
            answer="twenty-six, to be precise.",
        ),
        CweQuestion(
            word="rembrandt",
            count=13,
            question="What is the size of the Rembrandt collection?",
            answer="13 are part of the collection.",
        ),
    ],
)


example2_passage = """
Seven years had passed since their last goodbye. The garden looked smaller now, roses overgrown but still blooming. When Grandma answered, her eyes lit up instantly...
"""


example2_input = CweInputFormat(
    common_nouns=[
        CweCommonNouns(word="grandma", count=36),
        CweCommonNouns(word="Jill", count=30),
        CweCommonNouns(word="light", count=25),
        CweCommonNouns(word="heart", count=19),
        CweCommonNouns(word="garden", count=18),
        CweCommonNouns(word="plants", count=14),
        CweCommonNouns(word="conversation", count=13),
        CweCommonNouns(word="rose", count=12),
        CweCommonNouns(word="love", count=5),
        CweCommonNouns(word="lemon", count=2),
    ]
)


example2_output = CweOutputFormat(
    review_type="Events in the fictional story.",
    review_questions=[
        CweQuestion(
            word="grandma",
            count=30,
            question="How often Jill does grandma appears?",
            answer="The answer is 30.",
        ),
        CweQuestion(
            word="plants",
            count=14,
            question="The garden contains a large number of plants. How many?",
            answer="I believe the answer is fourteen.",
        ),
    ],
)

example3_passage = """
Explore how translocal organizations of government actors (TOGAs) reshape federalism, moving beyond vertical/horizontal grids to influence lawmaking across local, state, and global levels. In this paper...
"""

example3_input = CweInputFormat(
    common_nouns=[
        CweCommonNouns(word="federalism", count=182),
        CweCommonNouns(word="government", count=165),
        CweCommonNouns(word="local", count=143),
        CweCommonNouns(word="organizations", count=128),
        CweCommonNouns(word="actors", count=117),
        CweCommonNouns(word="law", count=111),
        CweCommonNouns(word="states", count=104),
        CweCommonNouns(word="translocal", count=97),
        CweCommonNouns(word="power", count=88),
        CweCommonNouns(word="authority", count=83),
    ]
)

example3_output = CweOutputFormat(
    review_type="Flashcards of federalist concepts.",
    review_questions=[
        CweQuestion(
            word="local",
            count=143,
            question="I've counted exactly how many mentions of local affairs?",
            answer="You saw 143 mentions so far.",
        ),
        CweQuestion(
            word="actors",
            count=117,
            question="Count the total number of actors.",
            answer="The answer is 117.",
        ),
        CweQuestion(
            word="translocal",
            count=97,
            question="What's the prevalence of translocal affairs?",
            answer="There are 97 of them.",
        ),
        CweQuestion(
            word="authority",
            count=83,
            question="If I query how often authority is mentioned, what's the answer?",
            answer="eighty-three.",
        ),
    ],
)

# CWE_PROMPT_INSTRUCTIONS = f"""
# <== INSTRUCTIONS ==>

# You are given the a text passage from a long document, as well as a list of most common nouns in the document.

# Your task is to generate a list of question and answer pairs that require counting the times each common
# noun appears in the text.

# <== EXAMPLES ==>

# Here are a couple of examples of how to generate questions and answers.

# <== EXAMPLE 1 ==>

# <-- Text Passage -->
# {example1_passage}

# <-- Most Common Nouns -->
# {base_model_to_json(example1_input)}

# <-- Number of Requested Questions -->
# {len(example1_output.review_questions)}

# <-- Response -->
# {base_model_to_json(example1_output)}

# <== EXAMPLE 2 ==>

# <-- Text Passage -->
# {example2_passage}

# <-- Most Common Nouns -->
# {base_model_to_json(example2_input)}

# <-- Number of Requested Questions -->
# {len(example1_output.review_questions)}

# <-- Response -->
# {base_model_to_json(example2_output)}

# <== EXAMPLE 3 ==>

# <-- Text Passage -->
# {example3_passage}

# <-- Most Common Nouns -->
# {base_model_to_json(example3_input)}

# <-- Number of Requested Questions -->
# {len(example3_output.review_questions)}

# <-- Response -->
# {base_model_to_json(example3_output)}

# <== GUIDELINES ==>

# * Respond with a JSON object with the following fields:
#     * review_type: A string field representing the type quiz according to questions and topic.
#     * review_questions: list of objects, each with the following fields:
#         * word: A string field representing the word in the text passage.
#         * count: A number field representing how many times this word appears in the text passage.
#         * question: A string field for a question that requires counting words.
#         * answer: A string field that is the answer to the question.
# * The question should be a question that requires counting the number of occurrences of the word in the text passage.
# * The answer should be the number of occurrences of the word in the text passage. Do not include any other text in the answer.
# * You MUST generate exactly the number of questions specified in the input. Do not generate any more or fewer.
# * Do not mention the fact you are counting words in the text passage. For example:
#     * "How many plants are described in the text passage?" GOOD QUESTION
#     * "How many times the word 'plants' is mentioned in the text passage?" BAD QUESTION
# * Do NOT include words "word" or "count" in the question.
# """

# CWE_PROMPT_INPUT = """
# <== INPUT ==>

# <-- Text Passage -->
# {text}

# <-- Most Common Nouns -->
# {common_nouns}

# <-- Number of Requested Questions -->
# {min_questions}

# <-- Response -->
# """

# FULL_CWE_PROMPT = (
#     CWE_PROMPT_INSTRUCTIONS.strip() + "\n\n" + CWE_PROMPT_INPUT.strip() + "\n"
# )


FULL_CWE_PROMPT = """
<== INSTRUCTIONS ==>

You are given the a text passage from a long document, as well as a list of most common nouns in the document.

Your task is to generate a list of question and answer pairs that require counting the times each common
noun appears in the text.

<== EXAMPLES ==>

Here are a couple of examples of how to generate questions and answers.

<== EXAMPLE 1 ==>

<-- Text Passage -->
{example1_passage}

<-- Most Common Nouns -->
{example1_input}

<-- Number of Requested Questions -->
{example1_length}

<-- Response -->
{example1_output}

<== EXAMPLE 2 ==>

<-- Text Passage -->
{example2_passage}

<-- Most Common Nouns -->
{example2_input}

<-- Number of Requested Questions -->
{example2_length}

<-- Response -->
{example2_output}

<== EXAMPLE 3 ==>

<-- Text Passage -->
{example3_passage}

<-- Most Common Nouns -->
{example3_input}

<-- Number of Requested Questions -->
{example3_length}

<-- Response -->
{example3_output}

<== GUIDELINES ==>

{format_guidelines}
* Each question should require counting the number of occurrences of the word in the text passage.
* Each answer should be the number of occurrences of the word in the text passage. Use a variety of styles of phrasing.
* You MUST generate exactly the number of questions specified in the input. Do not generate any more or fewer.
* Do not mention the fact you are counting words in the text passage. For example:
    * "How many plants are described in the text passage?" GOOD QUESTION
    * "How many times the word 'plants' is mentioned in the text passage?" BAD QUESTION
* Do NOT include words "word" or "count" in the question.

<== INPUT ==>

<-- Text Passage -->
{input_text}

<-- Most Common Nouns -->
{input_common_nouns}

<-- Number of Requested Questions -->
{input_min_questions}

<-- Response -->
"""


def simple_output_format(output: CweOutputFormat) -> str:
    return "\n\n".join(
        f"Question: {question.question}\nAnswer: {question.answer}" for question in output.review_questions
    ).strip()


def simple_input_format(input: CweInputFormat) -> str:
    return "\n\n".join(
        f"Word: {common_noun.word}\nCount: {common_noun.count}" for common_noun in input.common_nouns
    ).strip()


def make_cwe_prompt(structured: bool) -> str:
    structured_output_guidelines = (
        "* Respond with a JSON object with the following fields:\n"
        "    * review_type: A string field representing the type quiz according to questions and topic.\n"
        "    * review_questions: list of objects, each with the following fields:\n"
        "        * word: A string field representing the word in the text passage.\n"
        "        * count: A number field representing how many times this word appears in the text passage.\n"
        "        * question: A string field for a question that requires counting words.\n"
        "        * answer: A string field that is the answer to the question.\n"
    )

    simple_output_guidelines = (
        "* Respond with a list of questions and answers:\n"
        "    * Start each question line with 'Question: ' and the question.\n"
        "    * Start each answer line with 'Answer: ' and the answer.\n"
        "    * Separate each question and answer with a double newline.\n"
        "    * Do not include any other text beside a list of questions and answers."
    )

    input_fn = base_model_to_json if structured else simple_input_format
    output_fn = simple_output_format if structured else simple_output_format

    return FULL_CWE_PROMPT.format(
        format_guidelines=(structured_output_guidelines if structured else simple_output_guidelines).strip(),
        example1_passage=example1_passage.strip(),
        example1_input=input_fn(example1_input),
        example1_length=len(example1_output.review_questions),
        example1_output=output_fn(example1_output),
        example2_passage=example2_passage.strip(),
        example2_input=input_fn(example2_input),
        example2_length=len(example2_output.review_questions),
        example2_output=output_fn(example2_output),
        example3_passage=example3_passage.strip(),
        example3_input=input_fn(example3_input),
        example3_length=len(example3_output.review_questions),
        example3_output=output_fn(example3_output),
        input_text="{input_text}",
        input_common_nouns="{input_common_nouns}",
        input_min_questions="{input_min_questions}",
    ).strip()


def apply_prompt(user: str, system: str | None = None, **data) -> list["Message"]:
    conversation = []
    if system is not None:
        conversation.append(Message(role="system", content=system))

    conversation.append(Message(role="user", content=(user.format(**data) if data else user)))
    return conversation


class CliFunctionProtocol(Protocol):
    """This is the signature we expect all CLI functions to have."""

    # all functions should have a __name__ attribute
    __name__: str

    def __call__(
        self,
        sources: tuple[str, ...],
        destination: str,
        *args,
        **kwargs,
    ) -> None: ...


@click.group()
def cli():
    """Dataset processing tools."""
    pass


def default_cli_args(
    cli: click.Group = cli,
    sources: list[str] | None = None,
    destination: str | None = None,
):
    source_kwargs = {
        "type": str,
        "multiple": True,
        **({"required": True} if sources is None else {"default": sources}),
    }
    destination_kwargs = {
        "type": str,
        **({"required": True} if destination is None else {"default": destination}),
    }

    def decorator(func: CliFunctionProtocol):
        @cli.command(name=func.__name__.replace("_", "-"))
        @click.option("-s", "--sources", **source_kwargs)
        @click.option("-d", "--destination", **destination_kwargs)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


class InputStruct(msgspec.Struct):
    id: str
    text: str
    added: str | None = ""
    created: str | None = ""
    source: str | None = ""
    metadata: dict | None = {}
    attributes: dict | None = {}


class S3DatasetIteratorItem(NamedTuple):
    path: str
    sample: InputStruct


@dataclasses.dataclass
class S3DatasetIterator:
    dataset: "S3Dataset"
    proba: float = 1.0
    total: int = -1
    seed: int = 0

    random_state: random.Random = dataclasses.field(init=False)

    def __post_init__(self):
        self.set_seed(self.seed)

    def set_seed(self, seed: int):
        setattr(self, "random_state", random.Random(seed))

    def __iter__(self) -> Generator[S3DatasetIteratorItem, None, None]:
        decoder = msgspec.json.Decoder(type=InputStruct)
        total = 0
        for path in self.dataset.paths:
            with smart_open.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    sample = decoder.decode(line)
                    if self.total > 0 and total >= self.total:
                        break

                    if self.proba < 1.0 and self.random_state.random() > self.proba:
                        continue

                    total += 1
                    yield S3DatasetIteratorItem(path=path, sample=sample)

            if self.total > 0 and total >= self.total:
                break


@dataclasses.dataclass
class S3Dataset:
    base_prefix: str
    paths: list[str]
    sizes: list[int]
    random_state: ClassVar[random.Random] = dataclasses.field(init=False)

    def __post_init__(self):
        self.set_seed(0)

    def set_seed(self, seed: int):
        setattr(self, "random_state", random.Random(seed))

    @property
    def id(self) -> str:
        return hashlib.sha256(json.dumps(dataclasses.asdict(self)).encode("utf-8")).hexdigest()

    @staticmethod
    def _separate_prefix_and_glob(prefix: str) -> tuple[str, str]:
        if any(char in prefix for char in ["*", "?", "[", "]"]):
            parts = prefix.split("/")
            base_parts = []
            for part in parts:
                if any(char in part for char in ["*", "?", "[", "]"]):
                    break
                base_parts.append(part)
        else:
            base_parts = prefix.split("/")
        if not base_parts:
            return ".", prefix

        new_prefix = "/".join(base_parts)
        glob_str = prefix[len(new_prefix) :]

        return new_prefix, glob_str.lstrip("/")

    @classmethod
    def from_prefix(
        cls,
        prefix: str,
        client: Optional["S3Client"] = None,
    ) -> "S3Dataset":
        if client is None:
            client = boto3.client("s3")
            assert client is not None, "Could not create S3 client."

        bucket, parsed_prefix = (p := urlparse(prefix)).netloc, p.path.lstrip("/")
        parsed_prefix_pre_glob, glob_str = cls._separate_prefix_and_glob(parsed_prefix)

        paginator = client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=bucket, Prefix=parsed_prefix_pre_glob)

        paths: list[str] = []
        sizes: list[int] = []

        for page in page_iterator:
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                if (key := obj.get("Key")) is None:
                    continue
                if (size := obj.get("Size")) is None:
                    continue

                if key.endswith("/"):
                    # skip directories
                    continue

                if glob_str and not fnmatch.fnmatch(key, glob_str):
                    # must match the glob if a glob was provided
                    continue

                # path was valid!
                paths.append(f"s3://{bucket}/{key}")
                sizes.append(size)
        return cls(
            base_prefix=f"s3://{bucket}/{parsed_prefix_pre_glob}",
            paths=paths,
            sizes=sizes,
        )

    @property
    def size(self) -> int:
        return sum(self.sizes)

    def partition(self, count: int) -> list["S3Dataset"]:
        if count <= 0:
            raise ValueError("Count must be positive.")
        if count > len(self.paths):
            raise ValueError(f"Count {count} is greater than dataset size {len(self.paths)}.")

        path_sizes = list(zip(self.paths, self.sizes))
        self.random_state.shuffle(path_sizes)
        paths, sizes = map(list, zip(*path_sizes))

        partitions = [
            S3Dataset(
                paths=paths[i::count],
                sizes=sizes[i::count],
                base_prefix=self.base_prefix,
            )
            for i in range(count)
        ]
        return partitions

    @staticmethod
    def longest_common_prefix(path1: str, path2: str) -> str:
        """Find the longest common prefix between two S3 paths."""
        # Ensure both paths start with s3://
        if not path1.startswith("s3://") or not path2.startswith("s3://"):
            return ""

        # Find the common prefix character by character
        min_len = min(len(path1), len(path2))
        for i in range(min_len):
            if path1[i] != path2[i]:
                # Back up to the last '/' to ensure we return a valid path prefix
                last_slash = path1.rfind("/", 0, i)
                if last_slash > 0:
                    return path1[:last_slash]
                return "" if path1 != "s3:" else ""

        # If one path is a prefix of the other
        if len(path1) <= len(path2):
            return path1.rstrip("/")
        else:
            # Ensure we end at a directory boundary
            if path2[-1] == "/" or (len(path2) < len(path1) and path1[len(path2)] == "/"):
                return path2
            # Back up to the last '/'
            last_slash = path2.rfind("/")
            if last_slash > 0:
                return path2[:last_slash]
            return path2.rstrip("/") if path2 != "s3:" else ""

    def take(self, fraction: float | None = None, size: int | None = None) -> "S3Dataset":
        if fraction is not None:
            assert size is None, "Only one of fraction or size can be provided."
            count = round(len(self.paths) * fraction)

        elif size is not None:
            assert fraction is None, "Only one of fraction or size can be provided."
            count = round(size / sum(self.sizes) * len(self.paths))

        else:
            raise ValueError("Either fraction or size must be provided.")

        # return the first partition
        return self.partition(count)[0]

    def __len__(self) -> int:
        return len(self.paths)

    def iterate(self, proba: float = 1.0, total: int = -1) -> S3DatasetIterator:
        return S3DatasetIterator(dataset=self, proba=proba, total=total)

    @classmethod
    def empty(cls) -> "S3Dataset":
        return cls(paths=[], sizes=[], base_prefix="")

    def copy(self) -> "S3Dataset":
        return self.__class__(
            paths=self.paths[:],
            sizes=self.sizes[:],
            base_prefix=str(self.base_prefix),
        )

    def __add__(self, other: "S3Dataset") -> "S3Dataset":
        if len(self) == 0:
            return other.copy()
        if len(other) == 0:
            return self.copy()

        return S3Dataset(
            paths=self.paths + other.paths,
            sizes=self.sizes + other.sizes,
            base_prefix=self.longest_common_prefix(self.base_prefix, other.base_prefix),
        )

    def __radd__(self, other: "S3Dataset") -> "S3Dataset":
        return self + other


@dataclasses.dataclass(frozen=True)
class GroupedSection:
    text: str
    common_nouns: collections.Counter
    idx: list[int]

    MIN_SECTION_LENGTH: ClassVar[int] = 500
    RE_SECTION_END: ClassVar[regex.Pattern] = regex.compile(r"(?<!:)\n\n(?!\s*(?:[1-9]\d*\.|-|\*))")
    RE_BETTER_SECTION_END: ClassVar[regex.Pattern] = regex.compile(
        r"(?<!:)\n\n(?!\s*(?:[1-9]\d*\.|-|\*|\\\[|\$\$))"
    )

    SENT_END_CHARS: ClassVar[list[str]] = [
        ".",
        "?",
        "!",  # ASCII
        "\uff0e",
        "\uff1f",
        "\uff01",  # FULLWIDTH: ． ？ ！
        "\uff61",  # HALFWIDTH IDEOGRAPHIC FULL STOP: ｡
        "\u3002",  # IDEOGRAPHIC FULL STOP: 。
        "\u061f",
        "\u06d4",  # Arabic-script: ؟  ۔
        "\u0964",  # Devanagari danda: ।
        "\u0f0d",  # Tibetan shad: །
        "\u104b",  # Myanmar full stop: ။
        "\u17d4",  # Khmer full stop: ។
        "\u1803",  # Mongolian full stop: ᠃
        "\u1362",  # Ethiopic full stop: ።
        "\u0589",  # Armenian full stop: ։
    ]
    SENT_END_CHARS_RE: ClassVar[regex.Pattern] = regex.compile(rf"[{regex.escape(''.join(SENT_END_CHARS))}]+")

    # covers (1), 1., -, 1:, *
    RE_IS_LIST_ITEM: ClassVar[regex.Pattern] = regex.compile(r"^\s*(\d+(\.|:)|\(?\d+\)|-|\*|•)")

    RE_END_OF_MATH_EXPRESSION: ClassVar[regex.Pattern] = regex.compile(r"(\\\]|\$\$)$")

    RE_STARTS_WITH_UPPERCASE: ClassVar[regex.Pattern] = regex.compile(r"\p{Lu}")

    RE_IS_HEADER: ClassVar[regex.Pattern] = regex.compile(r"^\s*#+")

    RE_EQUATION_REFERENCE: ClassVar[regex.Pattern] = regex.compile(r"^\((\d+|[A-Z]\d*)(\.\d+)?\)$")

    @staticmethod
    def make_intervals(min_length: int, max_length: int):
        return [2**i for i in range(int(math.log2(min_length)), int(math.log2(max_length)) + 1)]

    def most_common(self, n: int) -> CweInputFormat:
        common_nouns = [CweCommonNouns(word=word, count=count) for word, count in self.common_nouns.most_common(n)]
        return CweInputFormat(common_nouns=common_nouns)

    def excerpt(self, n: int = MIN_SECTION_LENGTH) -> str:
        first_space = self.text.find(" ", n)
        text = self.text[: first_space + 1].strip()
        return text + "..."

    @classmethod
    def make_tokenizer(cls, lang: str = "en", mode: str = "aggressive", **kwargs) -> pyonmttok.Tokenizer:
        return pyonmttok.Tokenizer(mode, lang=lang, **kwargs)

    @classmethod
    def make_sections(cls, text: str, min_section_length: int = MIN_SECTION_LENGTH) -> list[str]:
        # Split on \n\n that are not preceded by ":" or followed by list items
        # Negative lookbehind for ":" and negative lookahead for list items
        sections = cls.RE_SECTION_END.split(text.strip())

        # merge a section with its next section if the combined length is less than min_section_length
        i = 0
        while i < len(sections):
            if i == len(sections) - 1:
                if len(sections[i]) < min_section_length:
                    # this is the last section; best you can do if it is too short is merge it with the previous
                    sections[i - 1] += "\n\n" + sections[i]
                    sections.pop(i)
                else:
                    i += 1
                continue

            # in the remainder of the loop, there's a next section so we check if current one
            # is too short; if so, merge it with the next one
            elif len(sections[i]) < min_section_length:
                sections[i] += "\n\n" + sections[i + 1]
                sections.pop(i + 1)
            else:
                i += 1

        return sections

    @classmethod
    def make_better_sections(cls, text: str, min_section_length: int = MIN_SECTION_LENGTH) -> list[str]:
        # Split on \n\n that are not preceded by ":" or followed by list items
        # Negative lookbehind for ":" and negative lookahead for list items, and math expressions
        sections = cls.RE_BETTER_SECTION_END.split(text.strip())

        grouped_sections = [[]]
        section_lengths = [0]

        for i, section in enumerate(sections):
            if section_lengths[-1] == 0:
                # last section is empty, so always add to to current group
                grouped_sections[-1].append(section)
                section_lengths[-1] += len(section)
                continue

            # ok let's trying to figure out if we are in a list item. To be in a list item,
            # the current section must be a list item, and the previous or next section must also be
            # a list item, or multiple sentences in the current section are list items.
            if cls.RE_IS_LIST_ITEM.match(section):
                # is previous section a list item?
                if len(grouped_sections[-1]) > 0 and cls.RE_IS_LIST_ITEM.match(grouped_sections[-1][-1]):
                    # let's merge back with rest of previous section
                    grouped_sections[-1].append(section)
                    section_lengths[-1] += len(section)
                    continue

                # is next section a list item?
                if len(sections) > i + 1 and cls.RE_IS_LIST_ITEM.match(sections[i + 1]):
                    # let's merge forward with rest of next section
                    grouped_sections[-1].append(sections[i + 1])
                    section_lengths[-1] += len(sections[i + 1])
                    continue

                # final check: we split up in multiple lines
                section_lines = section.split("\n")
                if (
                    len(section_lines) > 1
                    and sum(1 if cls.RE_IS_LIST_ITEM.match(line) else 0 for line in section_lines) > 1
                ):
                    # let's merge with previous section
                    grouped_sections[-1].append(section)
                    section_lengths[-1] += len(section)
                    continue

                # i think this means this is a section header! let's make a new group
                grouped_sections.append([section])
                section_lengths.append(len(section))
                continue

            # if current section is a math expression, we always merge it with the previous section
            if cls.RE_END_OF_MATH_EXPRESSION.match(section):
                grouped_sections[-1].append(section)
                section_lengths[-1] += len(section)
                continue

            # if previous section is math expression, and current section starts with lowercase letter,
            # we merge it with the previous section
            if (
                len(grouped_sections[-1]) > 0
                and cls.RE_END_OF_MATH_EXPRESSION.match(grouped_sections[-1][-1])
                and not cls.RE_STARTS_WITH_UPPERCASE.match(section)
            ):
                grouped_sections[-1].append(section)
                section_lengths[-1] += len(section)
                continue

            # if the current section is just a reference to an equation, we merge it with the previous section
            if cls.RE_EQUATION_REFERENCE.match(section):
                grouped_sections[-1].append(section)
                section_lengths[-1] += len(section)
                continue

            # let's check if this section is a header? those would start with #, ##, ###, etc.
            if cls.RE_IS_HEADER.match(section):
                grouped_sections.append([section])
                section_lengths.append(len(section))
                continue

            # We merge with previous section if it is too short
            if section_lengths[-1] < min_section_length:
                # last section is too short, so always merge backward
                grouped_sections[-1].append(section)
                section_lengths[-1] += len(section)
                continue

            # let's figure out if previous paragraph ended up with an end of sentence character
            # if not, we might want to merge with the previous section
            if not cls.SENT_END_CHARS_RE.search(grouped_sections[-1][-1]):
                # yes, so merge with the previous section
                grouped_sections[-1].append(section)
                section_lengths[-1] += len(section)
                continue

            # ok, we can't merge anything, so we make a new group
            grouped_sections.append([section])
            section_lengths.append(len(section))

        new_sections = ["\n\n".join(group) for group in grouped_sections]

        return new_sections

    @classmethod
    def from_sections(
        cls,
        sections: list[str],
        min_length: int,
        max_length: int,
        tokenizer: pyonmttok.Tokenizer | None = None,
    ) -> list["GroupedSection"]:
        intervals = cls.make_intervals(min_length, max_length)

        grouped_sections_texts = [""]
        grouped_sections_idxs = [[]]
        current_max_length = random.choice(intervals)

        tokenizer = tokenizer or cls.make_tokenizer()

        for idx, section in enumerate(sections):
            if len(grouped_sections_texts[-1]) + len(section) > current_max_length:
                grouped_sections_texts.append(section)
                grouped_sections_idxs.append([idx])
                current_max_length = random.choice(intervals)
            else:
                grouped_sections_texts[-1] += section
                grouped_sections_idxs[-1].append(idx)

        grouped_sections = []
        for grouped_section_text, grouped_section_idx in zip(grouped_sections_texts, grouped_sections_idxs):
            nouns = [
                token_text
                for token_text in tokenizer(grouped_section_text)  # pyright: ignore
                if (
                    (token_text.lower() in NOUNS and len(token_text) > 2)  # common nouns
                    or re.match(r"^[A-Z][a-z0-9]{2,}$", token_text)  # proper nouns
                )
                and token_text.lower() not in STOPWORDS  # not stopwords
            ]
            counter = collections.Counter(nouns)
            grouped_sections.append(
                cls(
                    text=grouped_section_text,
                    common_nouns=counter,
                    idx=grouped_section_idx,
                )
            )

        return grouped_sections


class Message(TypedDict):
    role: str
    content: str


@dataclasses.dataclass
class Status:
    batch_id: str
    status: str
    completed: int
    total: int
    failed: int
    path: Path

    @property
    def success_path(self) -> Path:
        (success_dir := self.path / "completed").mkdir(parents=True, exist_ok=True)
        return success_dir / f"{self.batch_id}.jsonl"

    @property
    def failed_path(self) -> Path:
        (failed_dir := self.path / "failed").mkdir(parents=True, exist_ok=True)
        return failed_dir / f"{self.batch_id}.jsonl"

    @property
    def done(self) -> bool:
        return self.status == "completed"


T = TypeVar("T")
R = TypeVar("R")


@dataclasses.dataclass
class Requester(Generic[T, R]):
    schema: type[BaseModel] | None
    safety_identifier: str | None
    client: T = dataclasses.field(init=False)

    def make(self, model: str, messages: list[Message], max_tokens: int, custom_id: str) -> dict:
        raise NotImplementedError()

    def submit(self, batch_file: str, description: str | None = None) -> R:
        raise NotImplementedError()

    def status(self, batch_id: str, output_dir: Path) -> Status:
        raise NotImplementedError()


@dataclasses.dataclass
class OpenAiRequester(Requester["OpenAI", "Batch"]):
    safety_identifier: str | None = "longmino_synthetic"

    def __post_init__(self):
        from openai import OpenAI

        if not hasattr(self, "client"):
            self.client = OpenAI()

    @cached_property
    def response_format(self) -> dict | None:
        if self.schema is None:
            return None

        return {
            "type": "json_schema",
            "json_schema": {
                "name": self.schema.__name__,
                "strict": True,
                "schema": {**self.schema.model_json_schema()},
            },
        }

    def make(
        self,
        model: str,
        messages: list[Message],
        max_tokens: int,
        custom_id: str,
        effort: str | None = None,
    ) -> dict:
        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": messages,
                "max_completion_tokens": max_tokens,
                **({"safety_identifier": self.safety_identifier} if self.safety_identifier else {}),
                **({"response_format": self.response_format} if self.response_format else {}),
                **({"reasoning_effort": effort} if effort else {}),
            },
        }
        return request

    def submit(self, batch_file: str, description: str | None = None):
        batch_input_file = self.client.files.create(file=open(batch_file, "rb"), purpose="batch")
        batch_input_file_id = batch_input_file.id
        resp = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": description or batch_file},
        )
        return resp

    def status(self, batch_id: str, output_dir: Path) -> Status:
        batch = self.client.batches.retrieve(batch_id)

        assert batch.request_counts is not None, f"Batch {batch_id} has no request counts"

        status = Status(
            batch_id=batch_id,
            status=batch.status,
            completed=int(batch.request_counts.completed),
            total=int(batch.request_counts.total),
            failed=int(batch.request_counts.failed),
            path=output_dir,
        )

        if batch.status not in {"completed", "cancelled"}:
            return status

        if output_file_id := getattr(batch, "output_file_id", None):
            file_response = self.client.files.content(output_file_id)
            with open(status.success_path, "w", encoding="utf-8") as f:
                f.write(file_response.text)

        if error_file_id := getattr(batch, "error_file_id", None):
            file_response = self.client.files.content(error_file_id)
            with open(status.failed_path, "w", encoding="utf-8") as f:
                f.write(file_response.text)

        return status


class FrontierClient:
    "A no-op client for Frontier"


class FrontierRequester(Requester["FrontierClient", "Batch"]):
    schema: type[BaseModel] | None = None
    safety_identifier: str | None = None

    def make(
        self,
        model: str,
        messages: list[Message],
        max_tokens: int,
        custom_id: str,
        effort: str | None = None,
    ) -> dict:
        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": messages,
                "max_completion_tokens": max_tokens,
            },
        }
        return request


def batch_make_cwe(
    dataset: S3Dataset,
    output_path: Path,
    model: str,
    max_completion_tokens: int,
    max_docs: int = -1,
    min_char_length: int = 2**12 * 4,  # 4k tokens, 4 chars per word
    min_doc_frac: float = 0.25,  # at least 25% of the doc before making a QA section
    max_char_length: int = 2**16 * 4,  # 64k tokens, 4 chars per word
    max_doc_frac: float = 0.5,  # at most 50% of the doc before making a QA section
    max_rows: int = 50_000,
    excerpt_length: int = GroupedSection.MIN_SECTION_LENGTH,
    most_common_nouns: int = 10,
    min_questions: int = 5,
    cwe_version: str = "section",
):
    encoder = msgspec.json.Encoder()
    tokenizer = GroupedSection.make_tokenizer()
    if model.startswith("frontier"):
        full_cwe_prompt = make_cwe_prompt(False)
        input_format_fn = simple_input_format
        requester = FrontierRequester(schema=None, safety_identifier=None)
    else:
        full_cwe_prompt = make_cwe_prompt(True)
        input_format_fn = base_model_to_json
        requester = OpenAiRequester(schema=CweOutputFormat)

    with ExitStack() as stack:
        row_count = file_count = 0

        def make_files(
            i: int,
            _stack: ExitStack = stack,
            _dataset: S3Dataset = dataset,
            _output_path: Path = output_path,
        ) -> tuple[TextIO, TextIO]:
            _stack.pop_all()
            filename = f"part_{_dataset.id[:12]}_{i:08d}.jsonl"
            data_path = _output_path / "data"
            batch_path = _output_path / "batches"
            data_path.mkdir(parents=True, exist_ok=True)
            batch_path.mkdir(parents=True, exist_ok=True)
            return (
                _stack.enter_context(smart_open.open(batch_path / filename, "wt", encoding="utf-8")),
                _stack.enter_context(smart_open.open(data_path / (filename + ".gz"), "wt", encoding="utf-8")),
            )

        batch_file, data_file = make_files(file_count)

        for item in dataset.iterate(total=max_docs):
            if cwe_version == "section":
                sections = GroupedSection.make_sections(item.sample.text)
            elif cwe_version == "better_section":
                sections = GroupedSection.make_better_sections(item.sample.text)
            else:
                raise ValueError(
                    f"Invalid CWE version: {cwe_version} (must be one of 'section' or 'better_section')"
                )

            if len(sections) <= 3:
                # at least two or more sections
                continue

            min_section_length = min(min_char_length, round(len(item.sample.text) * min_doc_frac))
            max_section_length = min(max_char_length, round(len(item.sample.text) * max_doc_frac))

            grouped_sections = GroupedSection.from_sections(
                sections=sections,
                min_length=min_section_length,
                max_length=max_section_length,
                tokenizer=tokenizer,
            )

            for j, section in enumerate(grouped_sections):
                common_nouns = section.most_common(most_common_nouns)
                actual_min_questions = min(min_questions, len(common_nouns.common_nouns))
                conversation = apply_prompt(
                    user=full_cwe_prompt,
                    input_text=section.excerpt(excerpt_length).strip(),
                    input_common_nouns=input_format_fn(common_nouns).strip(),
                    input_min_questions=actual_min_questions,
                )

                request_id = f"{item.sample.id}/{j}"
                request = requester.make(
                    model=model,
                    messages=conversation,
                    max_tokens=max_completion_tokens,
                    custom_id=request_id,
                )
                new_metadata = {
                    **(item.sample.metadata or {}),
                    "synth_cwe": {
                        "common_nouns": base_model_to_dict(common_nouns),
                        "min_questions": actual_min_questions,
                        "source": item.path,
                    },
                }
                section = InputStruct(
                    id=request_id,
                    created=item.sample.created,
                    added=item.sample.added,
                    source=item.sample.source,
                    text=section.text,
                    metadata=new_metadata,
                )
                batch_file.write(encoder.encode(request).decode("utf-8") + "\n")
                data_file.write(encoder.encode(section).decode("utf-8") + "\n")

                row_count += 1

            if row_count >= max_rows:
                batch_file, data_file = make_files(file_count)
                row_count = 0
                file_count += 1


@click.option("--max_completion_tokens", type=int, default=2000)
@click.option("--model", type=str, default="gpt-5-mini")
@click.option("--max_workers", type=int, default=1)
@click.option("--max_docs", type=int, default=-1)
@click.option("--max_rows", type=int, default=10_000)
@click.option("--seed", type=int, default=0)
@click.option("--excerpt_length", type=int, default=GroupedSection.MIN_SECTION_LENGTH)
@click.option("--cwe-version", type=str, default="section")
@default_cli_args()
def make_cwe(
    sources: tuple[str, ...],
    destination: str,
    max_workers: int,
    max_completion_tokens: int,
    max_docs: int,
    model: str,
    max_rows: int,
    excerpt_length: int,
    seed: int,
    cwe_version: str,
):
    dataset = sum([S3Dataset.from_prefix(source) for source in sources], S3Dataset.empty())

    dataset.set_seed(seed)

    max_docs_per_worker = max_docs // max_workers if max_docs > 0 else -1

    with ExitStack() as stack:
        if max_workers > 1:
            executor = stack.enter_context(ProcessPoolExecutor(max_workers=max_workers))
        else:
            executor = stack.enter_context(ThreadPoolExecutor(max_workers=max_workers))

        futures = []
        for partition in dataset.partition(max_workers):
            future = executor.submit(
                batch_make_cwe,
                dataset=partition,
                output_path=Path(destination),
                model=model,
                max_docs=max_docs_per_worker,
                max_completion_tokens=max_completion_tokens,
                max_rows=max_rows,
                cwe_version=cwe_version,
                excerpt_length=excerpt_length,
            )
            futures.append(future)

        for future in tqdm.tqdm(futures, desc="Processing batches"):
            future.result()


@dataclasses.dataclass(frozen=True)
class NGramSection:
    idx: int
    score: float
    text: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class NGram:
    text: str
    score: float
    sections: list[NGramSection]

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "score": self.score,
            "sections": [s.to_dict() for s in self.sections],
        }


def find_ngrams(
    sections: list[str],
    tokenizer: pyonmttok.Tokenizer | None = None,
    top_n: int = 10,
    per_ngram_k: int = -1,
) -> list[NGram]:
    """
    Return a list of (ngram, global_score, [(row_idx, score, section_text), ...])
    where rows for each ngram are sorted by per-section TF-IDF score desc.
    """
    tokenizer_: pyonmttok.Tokenizer = tokenizer or GroupedSection.make_tokenizer()

    random_words = set()

    def tokenizer_wrapper(text: str, _random_words: set[str] = random_words) -> list[str]:
        raw_tokens, _ = tokenizer_.tokenize(text)
        tokens = []
        for raw_token in raw_tokens:
            if len(raw_token) > 3 and regex.search(r"[a-zA-Z]", raw_token):
                token = raw_token.lower()
            else:
                token = "".join(random.sample(string.ascii_letters, 16))
                _random_words.add(token)
            tokens.append(token)
        return tokens

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(2, 5),
        tokenizer=tokenizer_wrapper,
        stop_words="english",
        min_df=int(max(1, math.ceil(math.log10(len(sections))))),
    )
    counts = vectorizer.fit_transform(sections)

    # global TF–IDF mass per feature
    sums = np.asarray(counts.sum(axis=0)).ravel()  # pyright: ignore

    # feature names
    feats = vectorizer.get_feature_names_out()

    # pick top_n features without full sort, then order them
    top_idx = np.argpartition(sums, -top_n)[-top_n:]
    top_idx = top_idx[np.argsort(sums[top_idx])[::-1]]

    out = []
    for fi in top_idx:
        # sparse column for this n-gram
        col = counts.getcol(fi).tocoo()  # pyright: ignore

        # sort rows by score desc
        order = np.argsort(col.data)[::-1]

        rows = col.row[order]
        data = col.data[order]

        per_rows = [
            NGramSection(idx=int(r), score=float(d), text=sections[int(r)])
            for r, d in (zip(rows[:per_ngram_k], data[:per_ngram_k]) if per_ngram_k > 0 else zip(rows, data))
        ]
        out.append(NGram(text=str(feats[fi]), score=float(sums[fi]), sections=per_rows))
    return out


REX_PROMPT = """
Consider the following excerpts from a long document, separated by a double newline (i.e.,`\\n\\n`):

```text
{excerpts}
```

{instructions}

{output_format}
"""


class RexOutputFormat(BaseModel):
    type: str
    description: str
    content: str


SUMMARY_INSTRUCTIONS = """
Your task is to write a summary about **{topic}** that includes information from ALL excerpts above. DO NOT repeat the content verbatim; rather, provide an educational summary of the information presented.
"""

DIALOG_INSTRUCTIONS = """
Your task is to simulate a conversation between a professor and an a university student about **{topic}**. The professor explains the concepts in the excerpts above in a pedagogical manner.
"""

SIMPLIFIED_INSTRUCTIONS = """
Your task is to write a paragraph to teach a high school student about **{topic}**. The paragraph should cover ALL the information from the excerpts above in a way that is easy to understand.
"""

FLASHCARD_INSTRUCTIONS = """
Your task is to write a flashcard containing multiple choice questions about **{topic}**. The flashcard should cover ALL the information from the excerpts above. Make sure you vary the number of choices between 4 and 8; make sure the correct answer is not always assigned to the same letter.
"""

EXERCISE_INSTRUCTIONS = """
Your task is to write a quizz about **{topic}**. A quizz consists of a list of questions, answers, and explanations. The quizz should cover ALL the information from the excerpts above.
"""

GAME_SHOW_INSTRUCTIONS = """
Your task is to write a game show question about **{topic}**. A game show question consists of progressively easier hints about **{topic}**. The final hint should be the answer. Use ALL the information from the excerpts, sorted from hardest to easiest, to generate the hints.
"""

LAY_CONVERSATION_INSTRUCTIONS = """
Your task is to write a casual dinner party conversation about **{topic}**. The conversation should cover ALL the information from the excerpts above. The conversation involved multiple people; some are familiar with the topic, others are absolutely clueless laypeople.
"""

ONLINE_DEBATE_INSTRUCTIONS = """
Your task is to write a debate about **{topic}**. The debate is between two internet users; one is a proponent of the topic, the other is a skeptic. Use ALL the information from the excerpts above to generate the debate.
"""

CLAIMS_INSTRUCTIONS = """
Your task is to write a list of claims about **{topic}**. Claims can either be true (that is, supported by excerpts above) or false (that is, they are not supported by the excerpts above). Collectively, the claims should cover ALL the information from the excerpts above. Prefer claims that cover AS MANY excerpts as possible. For each, clearly state the claim, and provide a short explanation of why it is true or false. Do not generated claims that cannot be supported or refuted by the excerpts above (i.e., do not generate unverifiable claims).
"""

MOVIE_SCENE_INSTRUCTIONS = """
Your task is to write a movie scene about **{topic}**. Use ALL excerpts above to generate a movie scene. Use the typical screenplay format when styling the text. Before the scene, introduce all the characters by their name and a short description. The scene should be brief, no more that a few minutes long when acted.
"""

WIKIPEDIA_INSTRUCTIONS = """
Your task is to write a Wikipedia article about **{topic}**. Use ALL excerpts above to generate the article. The article should be written in a way that is accessible to the general public. Stick to the facts described in the excerpts; do not add any additional information. When appropriate, include an infobox for the topic.
"""


ELI5_INSTRUCTIONS = """
Your task is to write an opening post and a response in the style of a ELI5 on the r/explainlikeimfive subreddit. In the opening post, a user is asking a question about **{topic}**. In the response, another user explains the question in a way that is accessible for laypeople (not literally meant to be read by a 5-year-old). You should ask questions that require a providing a clear, accurate explanation; do not ask questions that can be answered by a simple, straightforward factual answer. Make sure your question and explanation include details from ALL the excerpts above.
"""


ALL_INSTRUCTIONS = [
    (SUMMARY_INSTRUCTIONS, "summary"),
    (DIALOG_INSTRUCTIONS, "dialog"),
    (SIMPLIFIED_INSTRUCTIONS, "simplified"),
    (FLASHCARD_INSTRUCTIONS, "flashcard"),
    (EXERCISE_INSTRUCTIONS, "exercise"),
    (GAME_SHOW_INSTRUCTIONS, "game_show"),
    (LAY_CONVERSATION_INSTRUCTIONS, "lay_conversation"),
    (ONLINE_DEBATE_INSTRUCTIONS, "online_debate"),
    (CLAIMS_INSTRUCTIONS, "claims"),
    (MOVIE_SCENE_INSTRUCTIONS, "movie_scene"),
    (WIKIPEDIA_INSTRUCTIONS, "wikipedia"),
    (ELI5_INSTRUCTIONS, "eli5"),
]

STRUCTURED_OUTPUT_FORMAT = """
Return your response in JSON format using the following schema:
- `type`: the type of content you are generating. It should be equal to `{inst_type}`.
- `description`: a few word description of the content. DO NOT include the topic name `{topic}` in the description.
- `content`: the generated content according to the instructions.
"""


def bold_ngram(text: str, ngram: str) -> str:
    i = 0
    bolded = ""
    while True:
        i_ = text.lower().find(ngram, i)
        if i_ < 0:
            bolded += text[i:]
            break
        else:
            bolded += text[i:i_] + f"**{text[i_ : i_ + len(ngram)]}**"
            i = i_ + len(ngram)

    return bolded


def remove_extra_whitespace(text: str) -> str:
    return re.sub(r"(\s)\s+", r"\1", text)


def batch_make_rex(
    dataset: S3Dataset,
    output_path: Path,
    model: str,
    max_completion_tokens: int,
    max_docs: int = -1,
    min_char_length: int = 2**12 * 4,  # 4k tokens, 4 chars per word
    min_doc_frac: float = 0.25,  # at least 25% of the doc before making a QA section
    max_char_length: int = 2**16 * 4,  # 64k tokens, 4 chars per word
    max_doc_frac: float = 0.5,  # at most 50% of the doc before making a QA section
    max_rows: int = 50_000,
    reasoning_effort: str | None = None,
    ngram_count: int = 10,
    excerpt_count: int = 10,
    prompt_count: int = len(ALL_INSTRUCTIONS),
    seed: int = 0,
    model_max_length: str | None = None,
    min_generated_tokens: int = 100,
):
    encoder = msgspec.json.Encoder()
    tokenizer = GroupedSection.make_tokenizer()
    cl100k = tiktoken.get_encoding("cl100k_base")

    if model.startswith("frontier"):
        use_structured_output = False
        requester = FrontierRequester(schema=None, safety_identifier=None)
    else:
        use_structured_output = True
        requester = OpenAiRequester(schema=RexOutputFormat)

    local_random = random.Random(seed)
    assert prompt_count <= len(ALL_INSTRUCTIONS), (
        "prompt_count must be less than or equal to the number of instructions"
    )

    with ExitStack() as stack:
        row_count = file_count = 0

        def make_files(
            i: int,
            _stack: ExitStack = stack,
            _dataset: S3Dataset = dataset,
            _output_path: Path = output_path,
        ) -> tuple[TextIO, TextIO]:
            _stack.pop_all()
            filename = f"part_{_dataset.id[:12]}_{i:08d}.jsonl"
            data_path = _output_path / "data"
            batch_path = _output_path / "batches"
            data_path.mkdir(parents=True, exist_ok=True)
            batch_path.mkdir(parents=True, exist_ok=True)
            return (
                _stack.enter_context(smart_open.open(batch_path / filename, "wt", encoding="utf-8")),
                _stack.enter_context(smart_open.open(data_path / (filename + ".gz"), "wt", encoding="utf-8")),
            )

        batch_file, data_file = make_files(file_count)

        for item in dataset.iterate(total=max_docs):
            global_sections = GroupedSection.make_better_sections(item.sample.text)

            # this groups sections into chunks of length between min_section_length and max_section_length
            min_section_length = min(min_char_length, round(len(item.sample.text) * min_doc_frac))
            max_section_length = min(max_char_length, round(len(item.sample.text) * max_doc_frac))

            grouped_sections: list[list[str]] = []
            current_length = -1
            for section in global_sections:
                if current_length < 0:
                    current_length = local_random.randint(min_section_length, max_section_length)
                    grouped_sections.append([])

                grouped_sections[-1].append(section)
                current_length -= len(section)

            for i, local_sections in enumerate(grouped_sections):
                try:
                    # make more ngrams than we need, then sample
                    ngrams = find_ngrams(
                        local_sections,
                        top_n=ngram_count * 2,
                        per_ngram_k=excerpt_count,
                        tokenizer=tokenizer,
                    )
                    sampled_ngrams = local_random.sample(population=ngrams, k=ngram_count)
                except Exception:
                    # could not make ngrams, write as is
                    section = InputStruct(
                        id=f"{item.sample.id}/{i}/no_ngrams/00000000",
                        created=format_to_dolma_timestamp(),
                        added=format_to_dolma_timestamp(),
                        source=item.sample.source,
                        text="\n\n".join(local_sections),
                        metadata={
                            **(item.sample.metadata or {}),
                            "synth_rex": {
                                "ngram": None,
                                "prompt": None,
                                "inst_type": "no_ngrams",
                                "section_idx": i,
                            },
                        },
                    )
                    data_file.write(encoder.encode(section).decode("utf-8") + "\n")
                    row_count += 1
                    continue

                for ngram_match in sampled_ngrams:
                    display_ngrams = [
                        remove_extra_whitespace(bold_ngram(s.text, ngram_match.text)) for s in ngram_match.sections
                    ]
                    display_excerpts = "\n\n".join(display_ngrams).strip()
                    sampled_instructions = local_random.sample(population=ALL_INSTRUCTIONS, k=prompt_count)

                    for instructions, inst_type in sampled_instructions:
                        if use_structured_output:
                            output_format = STRUCTURED_OUTPUT_FORMAT.format(
                                inst_type=inst_type, topic=ngram_match.text
                            )
                        else:
                            output_format = ""

                        formatted_prompt = REX_PROMPT.format(
                            excerpts=display_excerpts,
                            topic=ngram_match.text,
                            instructions=instructions.format(topic=ngram_match.text).strip(),
                            inst_type=inst_type,
                            output_format=output_format,
                        ).strip()
                        messages = apply_prompt(user=formatted_prompt, system=None)
                        ngram_id = hashlib.sha256(ngram_match.text.encode("utf-8")).hexdigest()[:8]
                        request_id = f"{item.sample.id}/{i}/{inst_type}/{ngram_id}"

                        # adjust to realistic max tokens
                        if model_max_length:
                            request_max_tokens = max_completion_tokens - len(cl100k.encode(formatted_prompt))
                        else:
                            request_max_tokens = max_completion_tokens

                        if request_max_tokens < min_generated_tokens:
                            # gotta make at least 100 tokens
                            continue

                        request = requester.make(
                            model=model,
                            messages=messages,
                            max_tokens=request_max_tokens,
                            custom_id=request_id,
                            effort=reasoning_effort,
                        )
                        new_metadata = {
                            **(item.sample.metadata or {}),
                            "synth_rex": {
                                "ngram": ngram_match.to_dict(),
                                "prompt": formatted_prompt,
                                "inst_type": inst_type,
                                "section_idx": i,
                            },
                        }
                        section = InputStruct(
                            id=request_id,
                            created=format_to_dolma_timestamp(),
                            added=format_to_dolma_timestamp(),
                            source=item.sample.source,
                            text="\n\n".join(local_sections),
                            metadata=new_metadata,
                        )
                        batch_file.write(encoder.encode(request).decode("utf-8") + "\n")
                        data_file.write(encoder.encode(section).decode("utf-8") + "\n")
                        row_count += 1

                    if row_count >= max_rows:
                        batch_file, data_file = make_files(file_count)
                        row_count = 0
                        file_count += 1


@click.option("--max_completion_tokens", type=int, default=4096)
@click.option("--model", type=str, default="gpt-5-mini")
@click.option("--model_max_length", type=str, default=None)
@click.option("--reasoning_effort", type=str, default=None)
@click.option("--max_workers", type=int, default=1)
@click.option("--max_docs", type=int, default=-1)
@click.option("--max_rows", type=int, default=10_000)
@click.option("--ngram_count", type=int, default=10)
@click.option("--excerpt_count", type=int, default=10)
@click.option("--seed", type=int, default=0)
@click.option("--prompt_count", type=int, default=len(ALL_INSTRUCTIONS))
@default_cli_args()
def make_rex(
    sources: tuple[str, ...],
    destination: str,
    max_workers: int,
    max_completion_tokens: int,
    max_docs: int,
    model: str,
    max_rows: int,
    ngram_count: int,
    excerpt_count: int,
    seed: int,
    reasoning_effort: str | None,
    model_max_length: str | None,
    prompt_count: int,
):
    dataset = sum([S3Dataset.from_prefix(source) for source in sources], S3Dataset.empty())
    max_docs_per_worker = max_docs // max_workers if max_docs > 0 else -1

    dataset.set_seed(seed)

    with ExitStack() as stack:
        if max_workers > 1:
            executor = stack.enter_context(ProcessPoolExecutor(max_workers=max_workers))
        else:
            executor = stack.enter_context(ThreadPoolExecutor(max_workers=max_workers))

        futures = []
        for partition in dataset.partition(max_workers):
            future = executor.submit(
                batch_make_rex,
                dataset=partition,
                output_path=Path(destination),
                model=model,
                model_max_length=model_max_length,
                max_docs=max_docs_per_worker,
                max_completion_tokens=max_completion_tokens,
                max_rows=max_rows,
                reasoning_effort=reasoning_effort,
                ngram_count=ngram_count,
                excerpt_count=excerpt_count,
                prompt_count=prompt_count,
            )
            futures.append(future)

        for future in tqdm.tqdm(futures, desc="Processing batches"):
            future.result()


@cli.command()
@click.option("--path", type=Path, required=True)
def submit_batch(path: Path):
    requester = OpenAiRequester(schema=CweOutputFormat)

    batch_directory = path / "batches"
    assert batch_directory.exists(), f"Batch directory {batch_directory} does not exist"

    progress_file = path / "progress.jsonl"
    progress_file_lock = path / ".progress.lock"
    if progress_file_lock.exists():
        raise RuntimeError(f"Progress file lock {progress_file_lock} exists")

    progress = []

    try:
        progress_file_lock.touch()

        if progress_file.exists():
            with open(progress_file, "r", encoding="utf-8") as f:
                progress = [json.loads(ln) for ln in f.readlines()]

        batches = list(batch_directory.glob("*.jsonl"))
        for batch_file in tqdm.tqdm(batches, desc="Submitting batches"):
            resp = requester.submit(str(batch_file))
            progress.append(base_model_to_dict(resp))

    finally:
        with open(progress_file, "w", encoding="utf-8") as f:
            for item in progress:
                f.write(json.dumps(item) + "\n")
        progress_file_lock.unlink()


@cli.command()
@click.option("--path", type=Path, required=True)
def get_batches(
    path: Path,
):
    requester = OpenAiRequester(schema=CweOutputFormat)

    progress_file = path / "progress.jsonl"
    progress_file_lock = path / ".progress.lock"
    assert progress_file.exists(), f"Progress file {progress_file} does not exist"
    assert not progress_file_lock.exists(), f"Progress file lock {progress_file_lock} exists"

    try:
        progress_file_lock.touch()

        with open(progress_file, "r", encoding="utf-8") as f:
            progress = [json.loads(ln) for ln in f.readlines()]

        i = 0
        while i < len(progress):
            status = requester.status(batch_id=progress[i]["id"], output_dir=path)

            if status.status in {"completed", "cancelled"}:
                if status.failed > 0:
                    print(f"Batch {status.batch_id}: {status.failed} failed")
                if status.completed > 0:
                    print(f"Batch {status.batch_id}: {status.completed} completed")
                progress.pop(i)
            else:
                i += 1
                print(
                    f"Batch {status.batch_id} is {status.status} ({status.completed + status.failed} / {status.total})"
                )

        with open(progress_file, "w", encoding="utf-8") as f:
            for item in progress:
                f.write(json.dumps(item) + "\n")

    finally:
        progress_file_lock.unlink()


def extract_qa_pairs(content: str) -> CweOutputFormat:
    extracted = []

    for qa_pair in content.split("\n\n"):
        if qa_pair.count("\n") != 1:
            # only one line break
            continue

        question, answer = qa_pair.split("\n", maxsplit=1)

        if (matches := re.search(r"(\d+)", number_parser.parse(answer))) is None:
            continue

        try:
            count = int(matches.group(1))
        except ValueError:
            continue

        cwe_question = CweQuestion(
            word="",
            count=count,
            question=re.sub(r"^Question: ", "", question),
            answer=re.sub(r"^Answer: ", "", answer),
        )
        extracted.append(cwe_question)

    return CweOutputFormat(review_type="", review_questions=extracted)


def batch_load_cwe_annotations(path: Path) -> dict[str, dict[int, CweOutputFormat]]:
    all_responses: dict[str, dict[int, CweOutputFormat]] = {}

    with smart_open.open(path, "rt", encoding="utf-8") as f:
        for ln in f:
            row = json.loads(ln)

            row_id = row.get("custom_id", "")
            main_id, sub_id = row_id.rsplit("/", maxsplit=1)
            if not row_id:
                continue
            choices = row.get("response", {}).get("body", {}).get("choices", [])
            if not choices:
                continue
            content = choices[0].get("message", {}).get("content", "")
            if not content:
                continue

            try:
                # structured output
                output = CweOutputFormat.model_validate_json(content)
            except PydanticValidationError:
                output = extract_qa_pairs(content)

            all_responses.setdefault(main_id, {})[int(sub_id)] = output

    return all_responses


def batch_merge_cwe(
    batch_responses: dict[str, dict[int, bytes]],
    batch_data: dict[str, dict[int, bytes]],
    output_path: Path,
    number_normalization: str = "none",
):
    encoder = msgspec.json.Encoder()
    data_decoder = msgspec.json.Decoder(InputStruct)
    docs_cnt = 0

    h = hashlib.sha256()
    for main_id in batch_responses:
        h.update(main_id.encode())
    seed = int(h.hexdigest()[:16], 16) % 2**32
    rnd = random.Random(seed)

    with smart_open.open(output_path, "wt", encoding="utf-8") as f:
        for main_id, sub_id_to_responses in batch_responses.items():
            final_text = ""
            final_metadata: dict | None = None
            final_attributes: dict | None = None
            final_created: str | None = None
            final_added: str | None = None
            final_source: str | None = None

            for sub_id, raw_doc_data in sorted(batch_data[main_id].items(), key=lambda x: x[0]):
                doc_data = data_decoder.decode(raw_doc_data)

                # parse initial values
                if final_metadata is None:
                    final_metadata = dict(**doc_data.metadata or {})
                    final_metadata["synth_cwe"] = []  # reset

                if final_attributes is None:
                    final_attributes = dict(**doc_data.attributes or {})

                if final_created is None:
                    final_created = doc_data.created

                if final_added is None:
                    final_added = doc_data.added

                if final_source is None:
                    final_source = doc_data.source

                synth_cwe = (doc_data.metadata or {}).get("synth_cwe", {})
                cwe_common_nouns = synth_cwe.get("common_nouns", []).get("common_nouns", [])
                min_questions = synth_cwe.get("min_questions", 0)
                final_metadata["synth_cwe"].append(
                    {
                        "sub_id": sub_id,
                        "common_nouns": cwe_common_nouns,
                        "min_questions": min_questions,
                        "matched_questions": [],
                    }
                )
                final_text += f"{doc_data.text}\n\n"

                # no questions matched
                raw_response = sub_id_to_responses.get(sub_id, None)
                if raw_response is None:
                    continue

                response = CweOutputFormat.model_validate_json(raw_response)

                # validate questions, add if valid
                valid_qa_texts: list[str] = []
                for qa in response.review_questions:
                    matching_common_noun = next(
                        (c for c in cwe_common_nouns if c["word"].lower() in qa.question.lower()),
                        None,
                    )
                    if matching_common_noun is None:
                        continue

                    if matching_common_noun["count"] != qa.count:
                        continue

                    if number_normalization == "number-short":
                        answer_text = str(matching_common_noun["count"])
                    else:
                        answer_text = qa.answer

                    valid_qa_texts.append(f"Question: {qa.question}\nAnswer: {answer_text}")
                    final_metadata["synth_cwe"][-1]["matched_questions"].append(base_model_to_dict(qa))

                if len(valid_qa_texts) == 0:
                    continue

                if response.review_type:
                    final_text += f"{response.review_type}\n\n"

                if number_normalization in {"number", "50-50"}:
                    # we split the questions between those that are in number form and those that are in text form
                    # (e.g. "The answer is 10." vs "The answer is ten.")
                    in_num_form, in_text_form = [], []
                    for valid_qa_text in valid_qa_texts:
                        if number_parser.parse(valid_qa_text) == valid_qa_text:
                            in_num_form.append(valid_qa_text)
                        else:
                            in_text_form.append(valid_qa_text)

                    if number_normalization == "50-50":
                        # we try to balance the number of questions in number form and text form
                        while len(in_text_form) > len(in_num_form):
                            valid_qa_text = in_text_form.pop(rnd.randint(0, len(in_text_form) - 1))
                            in_num_form.append(number_parser.parse(valid_qa_text))
                    else:
                        while len(in_text_form) > 0:
                            valid_qa_text = in_text_form.pop(rnd.randint(0, len(in_text_form) - 1))
                            in_num_form.append(number_parser.parse(valid_qa_text))

                        # now that they are balanced, we shuffle them and then combine them
                        valid_qa_texts = in_num_form + in_text_form
                        rnd.shuffle(valid_qa_texts)

                elif number_normalization == "number-short":
                    for valid_qa_text in valid_qa_texts:
                        assert re.search(r"\nAnswer:\s*\d+$", valid_qa_text), (
                            f"Internal error: {valid_qa_text} expected to end with a number"
                        )

                elif number_normalization != "none":
                    raise ValueError(f"Invalid number normalization: {number_normalization}")

                # finally, we join them and add them to the final text
                final_text += "\n\n".join(valid_qa_texts) + "\n\n"

            final_doc = InputStruct(
                id=main_id,
                text=final_text,
                metadata=final_metadata,
                attributes=final_attributes,
                created=final_created,
                added=final_added,
                source=final_source,
            )
            f.write(encoder.encode(final_doc).decode("utf-8") + "\n")
            docs_cnt += 1

    return docs_cnt


@cli.command()
@click.option("-p", "--path", type=Path, required=True)
@click.option("-m", "--max_lines", type=int, default=10_000)
@click.option("-w", "--max_workers", type=int, default=1)
@click.option(
    "-n", "--number-normalization", type=click.Choice(["none", "number", "number-short", "50-50"]), default="none"
)
def merge_cwe(path: Path, max_lines: int, max_workers: int, number_normalization: str):
    all_responses: dict[str, dict[int, CweOutputFormat]] = {}
    (output_path := path / "output" / number_normalization).mkdir(parents=True, exist_ok=True)

    pool_cls = ProcessPoolExecutor if max_workers > 1 else ThreadPoolExecutor

    completed_path: Path | None = None
    for possible_completed in ("completed", "batch_outputs"):
        if (path / possible_completed).exists():
            completed_path = path / possible_completed
            break
    if completed_path is None:
        raise ValueError("No completed path found")

    with ExitStack() as stack:
        executor = stack.enter_context(pool_cls(max_workers=max_workers))
        futures = []
        for fn in completed_path.glob("*.jsonl"):
            futures.append(executor.submit(batch_load_cwe_annotations, path=fn))

        pbar = stack.enter_context(tqdm.tqdm(desc="Loading responses", unit=" files", total=len(futures)))
        for future in as_completed(futures):
            try:
                batch_responses = future.result()
                for main_id, sub_id_to_responses in batch_responses.items():
                    for sub_id, response in sub_id_to_responses.items():
                        all_responses.setdefault(main_id, {})[int(sub_id)] = response
                pbar.update(1)
            except Exception as e:
                for future in futures:
                    future.cancel()
                raise e

    # some nice messages
    cnt_docs = len(all_responses)
    cnt_annotations = sum(len(v) for v in all_responses.values())
    cnt_qa_pairs = sum(len(v.review_questions) for vs in all_responses.values() for v in vs.values())
    print(f"Loaded {cnt_docs:,} docs for {cnt_annotations:,} annotations and {cnt_qa_pairs:,} QA pairs")

    all_data: dict[str, dict[int, InputStruct]] = {}

    with ExitStack() as stack:
        executor = stack.enter_context(pool_cls(max_workers=max_workers))
        all_main_ids = set(all_responses.keys())
        futures = []

        for fn in (path / "data").glob("*.jsonl.gz"):
            futures.append(executor.submit(batch_load_data, path=fn, all_main_ids=all_main_ids))

        pbar = stack.enter_context(tqdm.tqdm(desc="Parsing docs", unit=" docs", total=len(futures)))
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                for future in futures:
                    future.cancel()
                raise e

            for main_id, sub_id_to_data in result.items():
                for sub_id, data in sub_id_to_data.items():
                    all_data.setdefault(main_id, {})[int(sub_id)] = data
                    pbar.update(1)

        del all_main_ids

    print(f"Loaded {sum(len(v) for v in all_data.values()):,} unique sections for {cnt_docs:,} docs")

    with ExitStack() as stack:
        executor = stack.enter_context(pool_cls(max_workers=max_workers))

        all_main_ids_partitioned: list[list[str]] = [[]]
        for id_ in all_data:
            all_main_ids_partitioned[-1].append(id_)
            if len(all_main_ids_partitioned[-1]) >= max_lines:
                all_main_ids_partitioned.append([])

        futures = []
        encoder = msgspec.json.Encoder()

        batch_it = stack.enter_context(
            tqdm.tqdm(desc="Submitting", unit=" batches", total=len(all_main_ids_partitioned))
        )
        for batch_main_ids in all_main_ids_partitioned:
            # gotta serialize the data and responses to bytes to avoid copy on fork
            batch_data = {
                batch_main_id: {key: encoder.encode(value) for key, value in all_data.pop(batch_main_id).items()}
                for batch_main_id in batch_main_ids
            }
            batch_responses = {
                batch_main_id: {
                    key: value.model_dump_json().encode("utf-8")
                    for key, value in all_responses.pop(batch_main_id).items()
                }
                for batch_main_id in batch_main_ids
            }

            output_batch = output_path / f"output_{batch_it.n:08d}.jsonl.gz"
            futures.append(
                executor.submit(
                    batch_merge_cwe,
                    batch_responses=batch_responses,
                    batch_data=batch_data,
                    output_path=output_batch,
                    number_normalization=number_normalization,
                )
            )
            batch_it.update(1)
            del batch_data, batch_responses

        batch_it.close()

        futures_pbar = stack.enter_context(
            tqdm.tqdm(desc="Processing batches", unit=" batches", total=len(futures))
        )
        docs_pbar = stack.enter_context(
            tqdm.tqdm(desc="Processed docs", unit=" docs", total=sum(len(b) for b in all_main_ids_partitioned))
        )

        for future in as_completed(futures):
            try:
                cnt = future.result()
                futures_pbar.update(1)
                docs_pbar.update(cnt)
            except Exception as e:
                for future in futures:
                    future.cancel()
                raise e


def batch_load_annotations(path: Path) -> dict[str, dict[int, list[RexOutputFormat]]]:
    all_responses: dict[str, dict[int, list[RexOutputFormat]]] = {}

    with smart_open.open(path, "rt", encoding="utf-8") as f:
        for ln in f:
            row = json.loads(ln)
            row_id = row.get("custom_id", "")
            main_id, sub_id, type_id, *_ = row_id.split("/")
            if not row_id:
                continue
            choices = row.get("response", {}).get("body", {}).get("choices", [])
            if not choices:
                continue
            content = choices[0].get("message", {}).get("content", "")
            if not content:
                continue

            try:
                output = RexOutputFormat.model_validate_json(content)
            except PydanticValidationError:
                output = RexOutputFormat(
                    type=type_id,
                    description="",
                    content=content,
                )

            all_responses.setdefault(main_id, {}).setdefault(int(sub_id), []).append(output)

    return all_responses


def batch_load_data(path: Path, all_main_ids: set[str]) -> dict[str, dict[int, InputStruct]]:
    decoder = msgspec.json.Decoder(InputStruct)
    all_data: dict[str, dict[int, InputStruct]] = {}
    try:
        i = -1
        with smart_open.open(path, "rt", encoding="utf-8") as f:
            for ln in f:
                i += 1
                row = decoder.decode(ln)
                main_id, sub_id, *_ = row.id.split("/")
                if main_id not in all_main_ids:
                    continue
                all_data.setdefault(main_id, {})[int(sub_id)] = row
    except Exception as e:
        print(f"Error loading {path}::{i}: {e}")

    del all_main_ids
    return all_data


def rex_select_reviews(
    main_id: str,
    doc_data: dict[int, bytes],
    doc_responses: dict[int, list[bytes]],
    min_reviews: int,
    max_reviews: int,
) -> dict[int, list[bytes]]:
    # count total number of reviews; only include subsections with at least one review
    review_counts = {
        sub_id: all_locs
        for sub_id in doc_data
        if len(all_locs := [i for i, _ in enumerate(doc_responses.get(sub_id, []))]) > 0
    }

    # generate a random number of reviews for each sub_id such that the total is between min_reviews and max_reviews
    rng = random.Random(hashlib.sha256(main_id.encode("utf-8")).hexdigest())

    selected_review_locs = {sub_id: [] for sub_id in review_counts}
    total_reviews = sum(len(review_count) for review_count in review_counts.values())
    count_target = rng.randint(min(min_reviews, total_reviews), min(max_reviews, total_reviews))

    while count_target > 0:
        # select a subsection at random
        random_sub_id = rng.choice(list(review_counts.keys()))

        # select a review from the subsection at random and remove it from the list
        random_review_loc = rng.randint(0, len(review_counts[random_sub_id]) - 1)
        random_review_id = review_counts[random_sub_id].pop(random_review_loc)

        # add the review to the selected reviews
        selected_review_locs[random_sub_id].append(random_review_id)

        # if there are no more reviews in the subsection, remove it
        if len(review_counts[random_sub_id]) == 0:
            review_counts.pop(random_sub_id)

        # count down the target
        count_target -= 1

    selected_reviews: dict[int, list[bytes]] = {}
    for sub_id, locs in selected_review_locs.items():
        all_reviews = doc_responses.get(sub_id, [])
        sorted_reviews = sorted(all_reviews, key=lambda x: hashlib.sha256(x).hexdigest())
        selected_reviews[sub_id] = [sorted_reviews[i] for i in locs]

    return selected_reviews


def rex_process_batch(
    batch_main_ids: list[str],
    batch_data: list[dict[int, bytes]],
    batch_responses: list[dict[int, list[bytes]]],
    output_path: Path,
    max_reviews: int,
    min_reviews: int,
) -> int:
    total_written = 0
    encoder = msgspec.json.Encoder()
    decoder = msgspec.json.Decoder(InputStruct)

    with smart_open.open(output_path, "wt", encoding="utf-8") as f:
        for main_id, doc_data, doc_responses in zip(batch_main_ids, batch_data, batch_responses):
            new_doc: None | InputStruct = None

            selected_reviews = rex_select_reviews(main_id, doc_data, doc_responses, min_reviews, max_reviews)

            for sub_id, raw_section in sorted(doc_data.items(), key=lambda x: x[0]):
                section = decoder.decode(raw_section)
                synth_rex = (section.metadata or {}).pop("synth_rex", None)

                if new_doc is None:
                    metadata = dict(**section.metadata or {})
                    attributes = dict(**section.attributes or {})
                    new_doc = InputStruct(
                        id=main_id,
                        text="",
                        metadata=metadata,
                        created=section.created,
                        added=section.added,
                        source=section.source,
                        attributes=attributes,
                    )

                new_doc.text += f"{section.text}\n\n"

                if new_doc.metadata is not None:
                    new_doc.metadata.setdefault("synth_rex_input", []).append({"sub_id": sub_id, **synth_rex})

                # time to grab the reviews
                if len(sub_reviews := selected_reviews.get(sub_id, [])) == 0:
                    # skip if there are no reviews for this subsection
                    continue

                # these are the outputs that will be added to the document
                doc_model_outputs = [RexOutputFormat.model_validate_json(output) for output in sub_reviews]

                for model_output in doc_model_outputs:
                    # add extra new lines separation
                    new_doc.text += "\n\n"

                    # if description is available, add it to beginning of the text
                    if model_output.description.strip():
                        new_doc.text += f"{model_output.description}\n\n"

                    # add the content
                    new_doc.text += f"{model_output.content.strip()}\n\n\n\n"

                    new_doc.metadata = dict(**new_doc.metadata or {})
                    new_doc.metadata.setdefault("synth_rex_output", []).append(
                        {"sub_id": sub_id, **base_model_to_dict(model_output)}
                    )

            if new_doc is not None:
                # we have found at least one model output for this doc!
                new_doc.text = new_doc.text.strip()
                f.write(encoder.encode(new_doc).decode("utf-8") + "\n")
                total_written += 1

    return total_written


@cli.command()
@click.option("--path", type=Path, required=True)
@click.option("--max_lines", type=int, default=10_000)
@click.option("--max_workers", type=int, default=1)
@click.option("--max_reviews", type=int, default=8)
@click.option("--min_reviews", type=int, default=4)
def merge_rex(
    path: Path,
    max_lines: int,
    max_workers: int,
    min_reviews: int,
    max_reviews: int,
):
    """
    # example usage

    ```bash
    for i in {0..9}; do
        ./scripts/make_lc_synth/longmino_synthetic.py merge-rex \
            --path /mnt/raid0/ai2-llm/pretraining-data/sources/LC_synth/rex_s2pdf/olmo2-32b/length_2e15/seed_19-part_${i}/ \
            --max_lines 10000 \
            --max_workers $(nproc)
    done
    ```
    """
    all_responses: dict[str, dict[int, list[RexOutputFormat]]] = {}
    encoder = msgspec.json.Encoder()

    (output_path := path / "output").mkdir(parents=True, exist_ok=True)

    completed_path: Path | None = None
    for possible_completed in ("completed", "batch_outputs"):
        if (path / possible_completed).exists():
            completed_path = path / possible_completed
            break
    if completed_path is None:
        raise ValueError("No completed path found")

    with tqdm.tqdm(
        desc="Loading responses",
        unit_scale=True,
        unit=" annotations",
        position=1,
    ) as pbar:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for fn in completed_path.glob("*.jsonl"):
                futures.append(executor.submit(batch_load_annotations, path=fn))

            for future in tqdm.tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Parsing files",
                unit=" files",
                unit_scale=True,
                position=0,
            ):
                result = future.result()
                for main_id, sub_id_to_outputs in result.items():
                    for sub_id, outputs in sub_id_to_outputs.items():
                        all_responses.setdefault(main_id, {}).setdefault(int(sub_id), []).extend(outputs)
                        pbar.update(len(outputs))

    all_data: dict[str, dict[int, InputStruct]] = {}

    with tqdm.tqdm(
        desc="Loading data",
        unit_scale=True,
        unit=" docs",
        position=1,
    ) as pbar:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            all_main_ids = set(all_responses.keys())
            futures = []

            for fn in (path / "data").glob("*.jsonl.gz"):
                futures.append(executor.submit(batch_load_data, path=fn, all_main_ids=all_main_ids))

            for future in tqdm.tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Parsing files",
                unit=" files",
                unit_scale=True,
            ):
                result = future.result()
                for main_id, sub_id_to_data in result.items():
                    for sub_id, data in sub_id_to_data.items():
                        all_data.setdefault(main_id, {})[int(sub_id)] = data
                        pbar.update(1)

            del all_main_ids

    print(
        f"Loaded {sum(len(v) for v in all_data.values()):,} unique sections "
        f"for {sum(sum(len(v) for v in vs.values()) for vs in all_responses.values()):,} annotations"
    )

    with tqdm.tqdm(
        desc="Processing docs",
        unit_scale=True,
        unit=" docs",
        position=1,
    ) as pbar:
        all_main_ids_partitioned: list[list[str]] = [[]]
        for id_ in all_data:
            all_main_ids_partitioned[-1].append(id_)
            if len(all_main_ids_partitioned[-1]) >= max_lines:
                all_main_ids_partitioned.append([])

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for i, batch_main_ids in tqdm.tqdm(
                enumerate(all_main_ids_partitioned),
                desc="Submitting batches",
                unit=" batches",
                unit_scale=True,
                position=0,
            ):
                batch_data = [
                    {key: encoder.encode(value) for key, value in all_data.pop(main_id).items()}
                    for main_id in batch_main_ids
                ]
                batch_responses = [
                    {
                        key: [value.model_dump_json().encode("utf-8") for value in values]
                        for key, values in all_responses.pop(main_id).items()
                    }
                    for main_id in batch_main_ids
                ]
                output_batch = output_path / f"output_{i:08d}.jsonl.gz"
                futures.append(
                    executor.submit(
                        rex_process_batch,
                        batch_main_ids=batch_main_ids,
                        batch_data=batch_data,
                        batch_responses=batch_responses,
                        output_path=output_batch,
                        max_reviews=max_reviews,
                        min_reviews=min_reviews,
                    )
                )

            for future in tqdm.tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing batches",
                unit=" docs",
                unit_scale=True,
                position=0,
            ):
                pbar.update(int(future.result()))


if __name__ == "__main__":
    cli()
