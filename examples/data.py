from pathlib import Path
from functools import partial
from typing import Dict, Optional, Tuple, List
from collections import defaultdict
from dataclasses import dataclass

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from omegaconf import OmegaConf, DictConfig


@dataclass
class DataModule:
    """
    Responsible for loading and tokenizing the dataset.
    Can handle local files or datasets from the Hugging Face Hub.

    If no dataset is specified, it will load a local file.
    Otherwise, it will load a dataset from the Hub.
    """


    cfg: DictConfig = None

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
            processor = LocalFileProcessor(
                cfg=self.cfg,
                tokenizer=self.tokenizer,
            )
        else:
            processor = name2processor[self.cfg.data.dataset_name](
                cfg=self.cfg,
                tokenizer=self.tokenizer,
            )

        self.raw_dataset, self.tokenized_dataset = processor.prepare_dataset()

        self.label2id = processor.get_label2id()
        self.id2label = {i: l for l, i in self.label2id.items()}

    def get_train_dataset(self, tokenized: bool = True) -> Dataset:
        ds_attr = "tokenized_dataset" if tokenized else "raw_dataset"
        return getattr(self, ds_attr)["train"]

    def get_eval_dataset(self, tokenized: bool = True) -> Dataset:
        ds_attr = "tokenized_dataset" if tokenized else "raw_dataset"

        if "validation" not in getattr(self, ds_attr):
            return None
        return getattr(self, ds_attr)["validation"]

    def get_test_dataset(self, tokenized: bool = True) -> Dataset:
        ds_attr = "tokenized_dataset" if tokenized else "raw_dataset"

        if "test" not in getattr(self, ds_attr):
            return None
        return getattr(self, ds_attr)["test"]


class GenericDatasetProcessor:
    """
    Can load any dataset from the Hub.
    """

    def __init__(self, cfg: DictConfig, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__()

        self.cfg = cfg
        self.tokenizer = tokenizer

    def set_label2id(self, train_dataset: Dataset) -> None:

        labels = train_dataset.unique(self.cfg.data.label_col)
        labels = sorted(labels)

        self.label2id = {label: i for i, label in enumerate(labels)}

    def get_label2id(self) -> Dict[str, int]:
        return self.label2id

    def prepare_dataset(self) -> Tuple[Dataset, Dataset]:
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


class PubHealthProcessor:
    """
    Data processor for the PUBHEALTH dataset.
    https://huggingface.co/datasets/health_fact
    """

    def __init__(self, cfg: DictConfig, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__()

        self.cfg = cfg
        self.tokenizer = tokenizer

    @staticmethod
    def get_label2id() -> Dict[str, int]:
        return {"false": 0, "mixture": 1,
                 "true": 2, "unproven": 3}

    def prepare_dataset(self) -> Tuple[Dataset, Dataset]:

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


class ArxivProcessor:
    
    """
    Data processor for the arxiv dataset.
    https://huggingface.co/datasets/ccdv/arxiv-classification

    Task: Given an arxiv paper, predict the subject area.
    """


    def __init__(self, cfg: DictConfig, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__()

        self.cfg = cfg
        self.tokenizer = tokenizer

    @staticmethod
    def get_label2id() -> Dict[str, int]:

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

    def prepare_dataset(self) -> Tuple[Dataset, Dataset]:

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
                label_col=self.cfg.data.label_col,
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

    def __init__(self, cfg: DictConfig, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__()

        self.cfg = cfg
        self.tokenizer = tokenizer

    def prepare_dataset(self) -> Tuple[Dataset, Dataset]:

        # get ending of file (csv, json, parquet)
        filetype = Path(self.cfg.data.data_files["train"][0]).suffix
        filetype = filetype.lstrip(".")  # suffix keeps period

        if filetype not in {"csv", "json", "parquet"}:
            raise ValueError(f"Files should end in 'csv', 'json', or 'parquet', not {filetype}.")
       
        data_files = OmegaConf.to_container(self.cfg.data.data_files)
        raw_dataset = load_dataset(filetype, data_files=data_files)

        # Limit the number of rows, if desired
        if self.cfg.data.n_rows is not None and self.cfg.data.n_rows > 0:
            for split in raw_dataset:
                max_split_samples = min(self.cfg.data.n_rows, len(raw_dataset[split]))
                raw_dataset[split] = raw_dataset[split].select(range(max_split_samples))

        cols = raw_dataset["train"].column_names
        self.set_label2id(raw_dataset["train"])
        
        raw_dataset = raw_dataset.map(
            partial(
                change_label_to_int,
                label2id=self.label2id,
                label_col=self.cfg.data.label_col,
            ),
            batched=True,
            num_proc=self.cfg.num_proc,
        )
        

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

    def set_label2id(self, train_dataset: Dataset) -> None:

        labels = train_dataset.unique(self.cfg.data.label_col)
        labels = sorted(labels)

        self.label2id = {label: i for i, label in enumerate(labels)}

    def get_label2id(self) -> Dict[str, int]:
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
) -> Dict[str, List[int]]:
    
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


def change_label_to_int(example: Dict[str, str], label2id: Dict[str, int], label_col: str):

    if isinstance(example[label_col], list):
        return {
            label_col: [label2id[l] for l in example[label_col]]
        }
    return {
            label_col: label2id[example[label_col]]
        }
                       



# map dataset name to processor
name2processor = defaultdict(lambda: GenericDatasetProcessor)

name2processor.update(
    {
        "health_fact": PubHealthProcessor,
        "ccdv/arxiv-classification": ArxivProcessor,
    }
)
