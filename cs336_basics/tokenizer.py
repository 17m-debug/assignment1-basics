from __future__ import annotations
from collections.abc import Iterable, Iterator
import regex as re


class tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = dict(vocab)
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.id_map = {v: k for k, v in vocab.items()}
        for token in self.special_tokens:
            if token.encode("utf-8") not in self.id_map:
                n = len(self.vocab)
                self.vocab[n] = token.encode("utf-8")
                self.id_map[token.encode("utf-8")] = n
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.pattern =  r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
                
        self.cache = {}
        

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        import pickle
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def get_pairs(self, word):
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def _bpe(self, word_bytes: bytes):
        word = tuple(bytes([b]) for b in word_bytes)
        pairs = self.get_pairs(word)
        if not pairs:
            return [self.id_map[word_bytes]]
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            pairs = self.get_pairs(word)
            if not pairs:
                break
        tokens = list(word)
        tokens = [self.id_map[token] for token in tokens]
        return tokens

    def encode(self, text: str):
        tokens = []
        pattern =  r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        if self.special_tokens:
            pat_special = "("  + "|".join(map(re.escape, sorted(self.special_tokens, key=len, reverse=True))) + ")"
            chunks = (re.split(pat_special, text))
        else:
            chunks = [text]
        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                tokens.append(self.id_map[chunk.encode("utf-8")])
                continue
            else:
                for word in re.findall(pattern, chunk):
                    word_bytes = word.encode('utf-8')
                    if word_bytes in self.cache:
                        tokens.extend(self.cache[word_bytes])
                    else:
                        word_tokens = self._bpe(word_bytes)
                        self.cache[word_bytes] = word_tokens
                        tokens.extend(word_tokens)
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:        
        for text in iterable:    
            yield from self.encode(text)
        

    def decode(self, ids: list[int]) -> str:
        text_bytes = b''.join(self.vocab[i] for i in ids)
        return text_bytes.decode('utf-8', errors='replace')