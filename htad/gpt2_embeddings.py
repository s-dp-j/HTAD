"""Dependency-light GPT-2 byte-pair tokenization for frozen WTE pooling.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


def _bytes_to_unicode() -> Dict[int, str]:
    values = list(range(ord("!"), ord("~") + 1))
    values += list(range(ord("¡"), ord("¬") + 1))
    values += list(range(ord("®"), ord("ÿ") + 1))
    encoded = list(values)
    offset = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            encoded.append(256 + offset)
            offset += 1
    return dict(zip(values, [chr(value) for value in encoded]))


def _pairs(word: Tuple[str, ...]):
    return set(zip(word[:-1], word[1:]))


class GPT2BPETokenizer:
    """The original GPT-2 byte-level BPE needed by the fixed English prompts."""

    def __init__(self, model_path: Path) -> None:
        with (model_path / "vocab.json").open("r", encoding="utf-8") as handle:
            self.encoder = json.load(handle)
        with (model_path / "merges.txt").open("r", encoding="utf-8") as handle:
            merges = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
        self.ranks = {tuple(value.split()): index for index, value in enumerate(merges)}
        self.byte_encoder = _bytes_to_unicode()
        self.cache: Dict[str, str] = {}
        # Equivalent to GPT2Tokenizer's pattern for the ASCII domain prompts.
        self.pattern = re.compile(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"
        )

    def _bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = _pairs(word)
        if not pairs:
            return token
        while True:
            pair = min(pairs, key=lambda value: self.ranks.get(value, float("inf")))
            if pair not in self.ranks:
                break
            first, second = pair
            merged: List[str] = []
            index = 0
            while index < len(word):
                try:
                    next_index = word.index(first, index)
                except ValueError:
                    merged.extend(word[index:])
                    break
                merged.extend(word[index:next_index])
                index = next_index
                if index < len(word) - 1 and word[index] == first and word[index + 1] == second:
                    merged.append(first + second)
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = tuple(merged)
            if len(word) == 1:
                break
            pairs = _pairs(word)
        value = " ".join(word)
        self.cache[token] = value
        return value

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for token in self.pattern.findall(text):
            encoded = "".join(self.byte_encoder[value] for value in token.encode("utf-8"))
            ids.extend(self.encoder[piece] for piece in self._bpe(encoded).split(" "))
        return ids


def gpt2_pooled_embeddings(model_path: Path, texts: Sequence[str]):
    """Return mean-pooled frozen WTE vectors and exact token IDs."""
    model_path = Path(model_path)
    tokenizer = GPT2BPETokenizer(model_path)
    state = torch.load(str(model_path / "pytorch_model.bin"), map_location="cpu")
    weight = state.get("wte.weight", state.get("transformer.wte.weight"))
    if weight is None:
        raise KeyError("GPT-2 checkpoint does not contain wte.weight")
    outputs, ids_by_text = [], []
    with torch.no_grad():
        for text in texts:
            ids = tokenizer.encode(text)
            if not ids:
                raise ValueError("GPT-2 BPE produced no tokens")
            outputs.append(weight[torch.tensor(ids, dtype=torch.long)].float().mean(dim=0).clone())
            ids_by_text.append(ids)
    return outputs, ids_by_text

