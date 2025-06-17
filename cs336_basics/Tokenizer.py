import regex as re

class Tokenizer():
    def __init__(self, vocab, merges, special_tokens=None):
        '''
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None
        '''
        self.vocab = vocab
        special_tokens = [] if not special_tokens else special_tokens
        special_tokens.sort(key=len, reverse=True)
        self.vocab_inverse = {v: k for k, v in self.vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens
        return 
      
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        '''
        vocab_filepath: str
        merges_filepath: str
        special_tokens: list[str] | None = None
        '''
        # TBD
        pass
        # vocab = open(vocab_filepath, 'rb').read()
        # merges = open(merges_filepath, 'rb').read()
        # return cls(vocab, merges, special_tokens)
    
    def encode(self, text:str) -> list[int]:
        # create document boundary by special tokens
        if self.special_tokens:
            delims   = self.special_tokens
            pattern  = "(" + "|".join(map(re.escape, delims)) + ")"
            parts = re.split(pattern, text)
        else:
            parts = [text]
      
        print(parts)
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
        print(encoded)
        return encoded

    def encode_iterable(self,iterable):
        pass

    def decode(self, ids):
        decoded_bytes = b''
        for id in ids:
            decoded_bytes += self.vocab[id]
        decoded = decoded_bytes.decode('utf-8', errors = 'replace')
        print(decoded)
        return decoded
    
