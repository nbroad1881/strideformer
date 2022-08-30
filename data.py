from pathlib import Path
from itertools import chain
from functools import partial
from typing import Dict
from collections import defaultdict
from dataclasses import dataclass

import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
import torch


@dataclass
class DataModule:

    cfg: Dict = None

    def __post_init__(self):
        if self.cfg is None:
            raise ValueError

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model.model_name_or_path,
        )

    def prepare_dataset(self) -> None:
        """
        Load in dataset and tokenize.

        If debugging, take small subset of the full dataset.
        """

        if self.cfg.data.dataset_name is None:
            processor = LocalFileProcessor
        else:
            processor = name2processor[self.cfg.data.dataset_name](
                cfg=self.cfg,
                tokenizer=self.tokenizer,
            )

        self.raw_dataset, self.tokenized_dataset = processor.prepare_dataset()

        self.label2id = processor.get_label2id()
        self.id2label = {i: l for l, i in self.label2id.items()}

    def get_train_dataset(self, tokenized: bool = True) -> datasets.Dataset:
        if tokenized:
            return self.tokenized_dataset["train"]
        return self.raw_dataset["train"]

    def get_eval_dataset(self, tokenized: bool = True) -> datasets.Dataset:
        if tokenized:
            if "validation" not in self.tokenized_dataset:
                return None
            return self.tokenized_dataset["validation"]

        if "validation" not in self.raw_dataset:
            return None

        return self.raw_dataset["validation"]

    def get_test_dataset(self, tokenized: bool = True) -> datasets.Dataset:
        if tokenized:
            if "test" not in self.tokenized_dataset:
                return None
            return self.tokenized_dataset["test"]

        if "test" not in self.raw_dataset:
            return None

        return self.raw_dataset["test"]


class StridedLongformerCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        """
        This expects to get examples from the dataset.
        Each row in the dataset will have
            input_ids [num_chunks, sequence_length]
            attention_mask [num_chunks, sequence_length]
            label [num_classes]

        The text should already be tokenized and padded to the documents longest sequence.


        """

        label_key = "label" if "label" in features[0] else "labels"

        ids = list(chain(*[x["input_ids"] for x in features]))
        mask = list(chain(*[x["attention_mask"] for x in features]))
        labels = [x[label_key] for x in features]

        longest_seq = max([len(x) for x in ids])

        ids = [x + [self.tokenizer.pad_token_id] * (longest_seq - len(x)) for x in ids]
        mask = [x + [0] * (longest_seq - len(x)) for x in mask]

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class GenericDatasetProcessor:
    """
    Can load any dataset from the Hub.
    """

    def __init__(self, cfg, tokenizer):
        super().__init__()

        self.cfg = cfg
        self.tokenizer = tokenizer

    def set_label2id(self, train_dataset):

        labels = train_dataset.unique(self.label_col)
        labels = sorted(labels)

        self.label2id = {label: i for i, label in enumerate(labels)}

    def get_label2id(self):
        return self.label2id

    def prepare_dataset(self):
        raw_dataset = load_dataset(
            self.cfg.data.dataset_name, self.cfg.data.dataset_config_name
        )

        # Limit the number of rows, if desired
        if self.cfg.data.n_rows is not None and self.cfg.data.n_rows > 0:
            for split in raw_dataset:
                max_split_samples = min(self.cfg.data.n_rows, len(raw_dataset[split]))
                raw_dataset[split] = raw_dataset[split].select(range(max_split_samples))

        cols = raw_dataset["train"].column_names

        tokenized_dataset = raw_dataset.map(
            partial(
                tokenize,
                tokenizer=self.tokenizer,
                max_length=self.cfg.data.max_seq_length,
                stride=self.cfg.data.stride,
                text_col=self.cfg.data.text_col,
                text_pair_col=self.cfg.data.text_pair_col,
                label_col=self.cfg.data.label_col,
            ),
            batched=self.cfg.data.stride in {None, 0},
            batch_size=self.cfg.data.map_batch_size,
            num_proc=self.cfg.num_proc,
            remove_columns=cols,
        )

        return raw_dataset, tokenized_dataset


class HealthFactProcessor:
    def __init__(self, cfg, tokenizer):
        super().__init__()

        self.cfg = cfg
        self.tokenizer = tokenizer

    @staticmethod
    def get_label2id():
        return {"false": 0, "mixture": 1, "true": 2, "unproven": 3}

    def prepare_dataset(self):
        raw_dataset = load_dataset(
            self.cfg.data.dataset_name, self.cfg.data.dataset_config_name
        )

        # Ignore examples that have bad labels
        raw_dataset = raw_dataset.filter(lambda x: x["label"] != -1)

        # Limit the number of rows, if desired
        if self.cfg.data.n_rows is not None and self.cfg.data.n_rows > 0:
            for split in raw_dataset:
                max_split_samples = min(self.cfg.data.n_rows, len(raw_dataset[split]))
                raw_dataset[split] = raw_dataset[split].select(range(max_split_samples))

        cols = raw_dataset["train"].column_names

        tokenized_dataset = raw_dataset.map(
            partial(
                tokenize,
                tokenizer=self.tokenizer,
                max_length=self.cfg.data.max_seq_length,
                stride=self.cfg.data.stride,
            ),
            batched=self.cfg.data.stride in {None, 0},
            batch_size=self.cfg.data.map_batch_size,
            num_proc=self.cfg.num_proc,
            remove_columns=cols,
        )

        return raw_dataset, tokenized_dataset


class ArxivProcessor:
    def __init__(self, cfg, tokenizer):
        super().__init__()

        self.cfg = cfg
        self.tokenizer = tokenizer

    @staticmethod
    def get_label2id():

        labels = [
            "math.AC",
            "cs.CV",
            "cs.AI",
            "cs.SY",
            "math.GR",
            "cs.CE",
            "cs.PL",
            "cs.IT",
            "cs.DS",
            "cs.NE",
            "math.ST",
        ]

        return {label: i for i, label in enumerate(labels)}

    def prepare_dataset(self):
        raw_dataset = load_dataset(
            self.cfg.data.dataset_name, self.cfg.data.dataset_config_name
        )

        # Limit the number of rows, if desired
        if self.cfg.data.n_rows is not None and self.cfg.data.n_rows > 0:
            for split in raw_dataset:
                max_split_samples = min(self.cfg.data.n_rows, len(raw_dataset[split]))
                raw_dataset[split] = raw_dataset[split].select(range(max_split_samples))

        cols = raw_dataset["train"].column_names

        tokenized_dataset = raw_dataset.map(
            partial(
                tokenize,
                tokenizer=self.tokenizer,
                max_length=self.cfg.data.max_seq_length,
                stride=self.cfg.data.stride,
            ),
            batched=self.cfg.data.stride in {None, 0},
            batch_size=self.cfg.data.map_batch_size,
            num_proc=self.cfg.num_proc,
            remove_columns=cols,
        )

        return raw_dataset, tokenized_dataset


class LocalFileProcessor:
    """
    Can load csv, json, or parquet files that are on local storage.
    """

    def __init__(self, cfg, tokenizer):
        super().__init__()

        self.cfg = cfg
        self.tokenizer = tokenizer

    def prepare_dataset(self):

        # get ending of file (csv, json, parquet)
        filetype = Path(self.cfg.data.data_files["train"][0]).suffix
        filetype = filetype.lstrip(".")  # suffix keeps period

        if filetype not in {"csv", "json", "parquet"}:
            raise ValueError(f"Files should end in 'csv', 'json', or 'parquet', not {filetype}.")

        raw_dataset = load_dataset(filetype, data_files=self.cfg.data.data_files)

        # Limit the number of rows, if desired
        if self.cfg.data.n_rows is not None and self.cfg.data.n_rows > 0:
            for split in raw_dataset:
                max_split_samples = min(self.cfg.data.n_rows, len(raw_dataset[split]))
                raw_dataset[split] = raw_dataset[split].select(range(max_split_samples))

        cols = raw_dataset["train"].column_names
        self.set_label2id(raw_dataset["train"])

        tokenized_dataset = raw_dataset.map(
            partial(
                tokenize,
                tokenizer=self.tokenizer,
                max_length=self.cfg.data.max_seq_length,
                stride=self.cfg.data.stride,
                text_col=self.cfg.data.text_col,
                label_col=self.cfg.data.label_col,
            ),
            batched=self.cfg.data.stride in {None, 0},
            batch_size=self.cfg.data.map_batch_size,
            num_proc=self.cfg.num_proc,
            remove_columns=cols,
        )

        return raw_dataset, tokenized_dataset

    def set_label2id(self, train_dataset):

        labels = train_dataset.unique(self.label_col)
        labels = sorted(labels)

        self.label2id = {label: i for i, label in enumerate(labels)}

    def get_label2id(self):
        """
        Must be called after `set_label2id`
        """
        return self.label2id


def tokenize(
    examples,
    tokenizer,
    max_length,
    stride=None,
    text_col="text",
    text_pair_col=None,
    label_col="label",
):
    tokenizer_kwargs = {
        "padding": False,
    }

    # If stride is not None, using sbert approach
    if stride is not None and stride > 0:
        tokenizer_kwargs.update(
            {
                "padding": True,
                "stride": stride,
                "return_overflowing_tokens": True,
            }
        )

    texts = [examples[text_col]]
    if text_pair_col is not None:
        texts.append(examples[text_pair_col])

    tokenized = tokenizer(
        *texts,
        truncation=True,
        max_length=max_length,
        **tokenizer_kwargs,
    )

    tokenized["labels"] = examples[label_col]

    return tokenized


# TODO: Make Default processor

# map dataset name to processor
name2processor = defaultdict(lambda: GenericDatasetProcessor)

name2processor.update(
    {
        "health_fact": HealthFactProcessor,
        "ccdv/arxiv-classification": ArxivProcessor,
    }
)
