import random
import numpy as np

from dataclasses import dataclass
from typing import Union, Optional

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import LlamaTokenizer
from transformers.utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@dataclass
class TLMDataCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None

        if labels is not None:
            max_label_length = self.max_length

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        return features["input_ids"], features["labels"]


class TLMDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        # beware streaming dataset does not support shuffle
        if "shuffle" in kwargs:
            del kwargs["shuffle"]
        super().__init__(*args, **kwargs)


def tokenize(data, tokenizer, max_length):
    ids = tokenizer(data["text"])["input_ids"]
    ids.append(tokenizer.eos_token_id)

    if len(ids) <= max_length:
        input_ids = ids[:-1]
        labels = ids[1:]
    else:
        start_idx = random.choice(range(0, len(ids) - max_length))
        input_ids = ids[start_idx : start_idx + max_length]
        labels = ids[start_idx + 1 : start_idx + max_length + 1]

    res = {"input_ids": input_ids, "labels": labels}

    return res


def prepare_data(dataset_name, tokenizer_name, max_length):
    dataset = load_dataset(
        dataset_name,
        split="train",
        streaming=True,
    )
    tokenizer = LlamaTokenizer.from_pretrained(
        tokenizer_name,
    )
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_dataset = dataset.map(
        lambda x: tokenize(x, tokenizer, max_length),
        remove_columns=["text"],
        batched=False,
    )
    data_collator = TLMDataCollator(
        tokenizer,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return tokenized_dataset, tokenizer, data_collator
