import regex as re
import json
from typing import Any, Iterable, Iterator
from pathlib import Path


from collections import Counter
import regex as re
from tqdm import tqdm
from collections import defaultdict

def pre_tokenize(data, special_token = []):
    # init vocab
    vocab = {b: bytes([b]) for b in range(256)}
    index = 256

    # use re to pre-tokenize
    data_list = [data]

    for sp in special_token:
        vocab[index] = sp.encode('utf-8')
        index += 1
        new_data_list = []
        for dat in data_list:
            new_data_list.extend(dat.split(sp))
        data_list = new_data_list

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokenized = []
    for data in data_list:
        data_pre_tokenized = re.finditer(PAT, data) # returns a list of words
        pre_tokenized.extend(data_pre_tokenized)

    counter = defaultdict(int)
    for tok in pre_tokenized:
        counter[tuple([i.to_bytes(1, 'big') for i in tok.group().encode("utf-8")])] += 1

    return counter, vocab

def convert(tok):
    if isinstance(tok, bytes):
        return tok
    # tok is a tuple of sub-tokens – flatten each of them
    return b"".join(convert(t) for t in tok)


def merge(pre_tokenized, max_size, vocab):
    # create an initial frequency table
    letter_table = Counter(pre_tokenized)
    merges = []
    merge_size = max_size - len(vocab)
    token_id = len(vocab)


    with tqdm(total=merge_size, desc="BPE Merges", unit="merge") as pbar:
        for _ in range(merge_size):
            # build the pair-frequency table
            pairs_table = {}
            for word, freq in letter_table.items():
                # print(word, freq, type(word))
                for i in range(len(word) - 1):
                    pair = word[i:i+2]
                    pairs_table[pair] = pairs_table.get(pair, 0) + freq

            # choose the most frequent pair
            if not pairs_table:
                break
            max_freq  = max(pairs_table.values())
            top_pairs = [k for k, v in pairs_table.items() if v == max_freq]
            top_p     = max(top_pairs)

            # create the new token
            new_token = convert(top_p[0]) + convert(top_p[1])
            merges.append((convert(top_p[0]), convert(top_p[1])))
            vocab[token_id] = new_token
            token_id += 1

            # rewrite every word using the new symbol
            new_letter_table = {}
            for word, freq in letter_table.items():
                new_word = []
                i = 0
                while i < len(word):
                    if i + 1 < len(word) and word[i:i+2] == top_p:
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_letter_table[tuple(new_word)] = freq

            letter_table = new_letter_table

            pbar.update(1)

    return vocab, merges

def BPE(input_path: str, data:str, vocab_size: int, special_tokens: list[str]):
    if not data:
        with open(input_path, encoding="utf-8") as f:
            data = f.read()

    # print(data[:100])
    pre_tokenized, vocab = pre_tokenize(data, special_tokens)
    # print('finished pretokentizing')
    # print(pre_tokenized[:100])
    # print('finished pretokentizing')
    vocab, merges = merge(pre_tokenized, vocab_size - len(special_tokens) +1, vocab)
    return vocab, merges




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
    
