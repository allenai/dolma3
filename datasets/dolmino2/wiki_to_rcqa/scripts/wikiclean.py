# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "spacy>=3.8,<3.9",
#   "regex",
#   "dolma==1.2.1",
#   "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
# ]
# ///

"""
This script does basic preprocessing to clean Wikipedia data and format into summaries and sections. Assumes input data fornatted via the Wikipedia preparation found here: https://github.com/allenai/dolma/blob/main/docs/getting-started.md

Author: Luca Soldaini
Usage: uv run scripts/wikiclean.py --documents <wikipedia_input_dir> --destination <output_dir> --processes <num_processes>
"""

from typing import NamedTuple
from dolma.core import data_types  # import DocResult, Document, Span
from dolma import add_tagger, BaseTagger
from dolma.cli.tagger import TaggerCli
from msgspec.structs import Struct
import spacy
import regex as re
import html
import argparse


class Section(NamedTuple):
    type: str
    text: str


class PatchedOutputSpec(Struct):
    id: str
    attributes: dict[str, list[tuple[int, int, str]]]
    source: str | None = None


class PatchedSpan(data_types.Span):
    def __init__(
        self,
        start: int,
        end: int,
        type: str,
        score: str,
        experiment: str | None = None,
        tagger: str | None = None,
    ):
        self.start = start
        self.end = end
        self.type = type
        self.score = score
        self.experiment = experiment
        self.tagger = tagger


data_types.OutputSpec = PatchedOutputSpec
data_types.Span = PatchedSpan


@add_tagger("wikiclean")
class WikicleanTagger(BaseTagger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])

    def clean_text(self, text: str) -> str:
        # replace &amp; with &
        text = html.unescape(text)

        # replace \xa0 with space
        text = text.replace("\xa0", " ")

        # remove space/punctuation right after opening parentheses
        text = re.sub(r"\([\p{Po}\p{Pc}\p{Pf}\p{Pd}\s]+([^\p{P}])", r"(\1", text)

        # remove space/punctuation right before closing parentheses
        text = re.sub(r"([^\p{P}])[\p{Po}\p{Pc}\p{Pf}\p{Pd}\s]+\)", r"\1)", text)

        # remove empty parentheses
        text = re.sub(r"\(\s*\)", "", text)

        # remove references
        text = re.sub(r"\[\d+\]", "", text)

        # remove other style of references
        text = re.sub(r"<ref.+?</ref>", "", text)

        # remove empty quotes
        text = text.replace('""', "").replace("''", "")

        # remove space before punctuation
        text = re.sub(r"(\w)\s([.;!?,])", r"\1\2", text)

        # remove multiple spaces
        text = re.sub(r"( )+", " ", text)

        return text.strip()

    def is_valid_page(self, title: str, text: str) -> bool:
        if not (text.endswith(".") or text.endswith("?") or text.endswith("!")):
            return False

        if text.count(" ") < 20:
            return False

        if len(text) < 500:
            return False

        if "List of" in title:
            return False

        if "(disambiguation)" in title:
            return False
        return True

    def group_output(
        self, doc_result: "data_types.DocResult"
    ) -> "data_types.TaggerOutputDictType":
        tagger_output: "data_types.TaggerOutputDictType" = {
            field: [] for field in self.defaults
        }
        for span in doc_result.spans:
            output = span.start, span.end, span.score
            tagger_output.setdefault(span.type, []).append(output)
        return tagger_output

    def predict(self, doc: "data_types.Document") -> "data_types.DocResult":
        title, text = doc.text.split("\n\n", 1)
        lines = text.split("\n")

        processed_lines: list[Section] = []

        if self.is_valid_page(title, text):
            processed_lines.append(Section(type="title", text=title))

            # split into sections
            for line in lines:
                line = self.clean_text(line)
                if line.count(" ") < 10 and line.endswith("."):
                    # not a possible section
                    # Use spacy to check if line contains verbs
                    doc_line = self.nlp(line)
                    has_verb = any(token.pos_ == "VERB" for token in doc_line)

                    if not has_verb:
                        # No verbs found, this is likely a title
                        processed_lines.append(
                            Section(type="header", text=line.strip(".").strip())
                        )
                        continue

                processed_lines.append(Section(type="text", text=line))

        # Remove adjacent headers, keeping only the last one
        filtered_lines: list[Section] = []
        i = 0
        while i < len(processed_lines):
            if processed_lines[i].type == "header":
                # Look ahead to see if there are more headers
                j = i
                while j < len(processed_lines) and processed_lines[j].type == "header":
                    j += 1
                # Add only the last header in the sequence
                filtered_lines.append(processed_lines[j - 1])
                i = j
            else:
                filtered_lines.append(processed_lines[i])
                i += 1

        processed_lines = filtered_lines

        # count the length of the text sections until first header
        summary_text = ""
        for section in processed_lines:
            if section.type == "header":
                break
            summary_text += section.text + "\n"
        summary_text = summary_text.strip()

        full_text = "\n".join(
            (
                section.text + "\n"
                if section.type == "title"
                else section.text
                if section.type == "text"
                else ""
            )
            for section in processed_lines
        )

        if len(summary_text) == 0:
            return data_types.DocResult(doc=doc, spans=[])

        init_span = data_types.Span(
            start=0, end=len(doc.text), type="summary", score=summary_text
        )
        full_span = data_types.Span(
            start=0, end=len(doc.text), type="full_text", score=full_text
        )

        return data_types.DocResult(doc=doc, spans=[init_span, full_span])


def main():
    parser = argparse.ArgumentParser()
    TaggerCli.make_parser(parser)
    args = parser.parse_args()
    args.taggers = ["wikiclean"]
    TaggerCli.run_from_args(args)


if __name__ == "__main__":
    main()
