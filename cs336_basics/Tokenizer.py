import regex as re
import json
from typing import Any, Iterable, Iterator
from pathlib import Path

class Tokenizer():
    def __init__(self, 
                 vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens: list[str] | None = None):
        
        self.vocab = vocab
        special_tokens = [] if not special_tokens else special_tokens
        special_tokens.sort(key=len, reverse=True)
        self.vocab_inverse = {v: k for k, v in self.vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens
        return 
    
    @classmethod
    def from_files(cls,
                   vocab_filepath: str | Path,
                   merges_filepath: str | Path,
                   special_tokens_filepath: str | None = None):

        raw_vocab: dict[str, str] = json.loads(Path(vocab_filepath).read_text("utf-8"))
        vocab: dict[int, bytes] = {
            int(k): v.encode("utf-8") for k, v in raw_vocab.items()
        }

        raw_merges: list[list[str]] = json.loads(Path(merges_filepath).read_text("utf-8"))
        merges: list[tuple[bytes, bytes]] = [
            (a.encode("utf-8"), b.encode("utf-8")) for a, b in raw_merges
        ]

        special_tokens: list[str] | None = None
        if special_tokens_filepath is not None:
            special_tokens = json.loads(Path(special_tokens_filepath).read_text("utf-8"))

        return cls(vocab, merges, special_tokens)
    
    def to_files(self,
                 vocab_filepath: str | Path,
                 merges_filepath: str | Path,
                 special_tokens_filepath: str | None = None) -> None:
        
        vocab_json: dict[str, str] = {
            str(idx): token.decode("utf-8") for idx, token in self.vocab.items()
        }
        Path(vocab_filepath).write_text(
            json.dumps(vocab_json, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        merges_json: list[list[str]] = [
            [a.decode("utf-8"), b.decode("utf-8")] for a, b in self.merges
        ]
        Path(merges_filepath).write_text(
            json.dumps(merges_json, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if special_tokens_filepath is not None:
            Path(special_tokens_filepath).write_text(
                json.dumps(self.special_tokens, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def encode(self, text:str) -> list[int]:
        # create document boundary by special tokens
        if self.special_tokens:
            delims   = self.special_tokens
            pattern  = "(" + "|".join(map(re.escape, delims)) + ")"
            parts = re.split(pattern, text)
        else:
            parts = [text]
      
        # encode each document chunk into list of ints
        encoded = []
        for chunk in parts:
            if not chunk:
                continue

            if chunk in self.special_tokens:
                encoded.append(self.vocab_inverse[chunk.encode('utf-8')])
                # print(f'found special token {chunk}')
                continue
            
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            pre_tokenized = re.finditer(PAT, chunk)
            for tok in pre_tokenized:
                word = [i.to_bytes(1, 'big') for i in tok.group().encode("utf-8")]
                # print(f'found word {word}')

                for l, r in self.merges:
                    i = 0
                    new_word = []
                    while i < len(word):
                        if i < len(word)-1 and l == word[i] and r == word[i+1]:
                            new_word.append(l+r)
                            i += 2
                        else:
                            new_word.append(word[i])
                            i += 1
                    word = new_word

                for w in word:
                    encoded.append(self.vocab_inverse[w])
        return encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            for t in self.encode(s):
                yield t


    def decode(self, ids):
        decoded_bytes = b''
        for id in ids:
            decoded_bytes += self.vocab[id]
        decoded = decoded_bytes.decode('utf-8', errors = 'replace')
        return decoded
    
