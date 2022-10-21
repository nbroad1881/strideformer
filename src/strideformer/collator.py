from itertools import chain
from typing import List, Dict

import torch
from transformers import PreTrainedTokenizerFast


class StrideformerCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerFast, max_chunks: int = 128):
        """
        Loads a collator designed for Strideformer.

        Args:
            tokenizer (`PreTrainedTokenizerFast`):
                The tokenizer that corresponds with the first model in Strideformer.
            max_chunks (`int`, *optional*, defaults to 128):
                The maximum number of chunks that can be passed to the first model.
                This is to limit OOM errors.

        """
        self.tokenizer = tokenizer
        self.max_chunks = max_chunks

    def __call__(self, features: List[Dict]):
        """
        Put features in a format that the model can use.

        Args:
            features (`List[Dict]`):
                The list will be as long as the batch size specified
                passed to the DataLoader.ffffffffffffffffffffffffffff
                Each element of features will have keys: input_ids, attention_mask, labels
                input_ids will be of shape [num_chunks, sequence_length]
                attention_mask will be of shape [num_chunks, sequence_length]
                label will be a single value if this is single_label_classification or regression
                It will be a list if multi_label_classification


        Returns:
            (dict): input_ids, attention_mask, labels to be passed to the model.
        """

        label_key = "label" if "label" in features[0] else "labels"

        ids = list(chain(*[x["input_ids"] for x in features]))
        mask = list(chain(*[x["attention_mask"] for x in features]))
        labels = [x[label_key] for x in features]

        longest_seq = max([len(x) for x in ids])

        ids = [x + [self.tokenizer.pad_token_id] * (longest_seq - len(x)) for x in ids]
        mask = [x + [0] * (longest_seq - len(x)) for x in mask]

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long)[: self.max_chunks, :],
            "attention_mask": torch.tensor(mask, dtype=torch.long)[
                : self.max_chunks, :
            ],
            "labels": torch.tensor(labels, dtype=torch.long),
        }
