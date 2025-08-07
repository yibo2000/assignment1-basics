# some code from lecture_01.py
# export PYTHONPATH=./

import json
import regex as re # not import re
import os
from typing import Iterable, Iterator
from collections import OrderedDict

class Tokenizer():
    """Abstract interface for a tokenizer."""
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError
    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError


class BPETokenizerParams:
    """
    All you need to specify a BPETokenizer.
    vocab: dict[int, bytes] The tokenizer vocabulary, a mapping 
        from int (token ID in the vocabulary) to bytes (token bytes).
    merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item
        is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
        <token2>. The merges should be ordered by order of creation.
    """
    vocab: dict[int, bytes]    # index -> bytes
    merges: list[tuple[bytes, bytes]]  # bytes (<token1>, <token2>)
    name: str # name of training set
    
    def __init__(self, _vocab, _merges, _name = None):
        self.vocab = _vocab
        self.merges = _merges
        self.name = _name


class BPETokenizer(Tokenizer):
    params: BPETokenizerParams
    special_tokens: list[str] | None
    vocab_b_to_idx: dict[bytes, int] # bytes -> index, for encoding
    merge_hash: dict[bytes, bytes] # bytes -> bytes

    """BPE tokenizer given a set of merges and a vocabulary."""
    def __init__(self, vocab, merges, special_tokens=None):
        self.params = BPETokenizerParams(vocab, merges)
        if(special_tokens != None): special_tokens.sort(key=len, reverse=True)
        self.special_tokens = special_tokens
        self.vocab_b_to_idx = {}
        for idx, bt in self.params.vocab.items():
            self.vocab_b_to_idx[bt] = idx
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        cls.params = load_bpe_from_file(vocab_filepath, merges_filepath)
        cls.special_tokens = special_tokens

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        # split the text based on special tokens
        if(self.special_tokens != None):
            pattern = '|'.join(re.escape(token) for token in self.special_tokens)
            split_text = re.split(f'({pattern})', text) # keep special tokens
        else:
            split_text = [text]

        # pre-tokenize
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        compiled_pat = re.compile(PAT)
        pre_tokens = []
        for te in split_text:
            if(self.special_tokens != None and te in self.special_tokens): pre_tokens += [te]
            else: pre_tokens += [ match.group(0) for match in compiled_pat.finditer(te) ]

        # encoding
        # process each pre-token 
        pretoken_to_res = {} # pre-tokens -> list[int]
        for token in set(pre_tokens):
            if(self.special_tokens != None and token in self.special_tokens): pretoken_to_res[token] = [self.vocab_b_to_idx[ token.encode("utf-8") ]]
            else:
                indices = []
                for c in token:
                    """
                    # rearrangement, e.g.:
                    # "🙃" -> [172, 253, 247, 225] -> [8582, 247, 225] # by gpt-2
                    # "🙃" bytes -> [240, 159, 153, 131]
                    # delta = [-68, 94, 94, 94]
                    
                    see `data_gym_to_mergeable_bpe_ranks` in https://github.com/openai/tiktoken/blob/main/tiktoken/load.py
                    the oracle BPETokenizer in tests/test_tokenizer.py (tiktoken.get_encoding("gpt2")) rearranges the token from 0 to 255:
                    from
                    ```
                    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)} # the implementation in lecture_01.py
                    ```
                    to
                    ```
                    rank_to_intbyte = [b for b in range(2**8) if chr(b).isprintable() and chr(b) != " "]
                    data_gym_byte_to_byte = {chr(b): b for b in rank_to_intbyte}
                    n = 0
                    for b in range(2**8):
                        if b not in rank_to_intbyte:
                            rank_to_intbyte.append(b)
                            data_gym_byte_to_byte[chr(2**8 + n)] = b
                            n += 1
                    assert len(rank_to_intbyte) == 2**8
                    ```
                    """
                    lst = list(map(int, c.encode("utf-8")))
                    if(len(lst) == 1): 
                        indices.append(self.vocab_b_to_idx[c.encode("utf-8")])
                        continue
                    for i in lst:
                        if( i >=33 and i <= 126 ): indices.append( i - 33 )
                        elif ( i >= 161 and i <= 172 ): indices.append(i-67)
                        elif ( i >= 174 and i <= 255 ): indices.append(i-68)
                        else: indices.append(i + 94)
                
                for t0, t1 in self.params.merges:
                    pair = (self.vocab_b_to_idx[t0], self.vocab_b_to_idx[t1])
                    indices = merge(indices, pair, self.vocab_b_to_idx[t0+t1])
                pretoken_to_res[token] = indices
        
        res = []
        for token in pre_tokens:
            res += pretoken_to_res[token]
        return res

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-eﬀicient tokenization of large files that we cannot directly load into memory.
        """
        res = []
        for text in iterable:
            res.extend(self.encode(text))
        return res
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        bytes_list = list(map(self.params.vocab.get, ids))  # @inspect bytes_list
        string = b"".join(bytes_list).decode("utf-8", errors='replace')  # @inspect string
        return string


def save_bpe(output_dir:str, params: BPETokenizerParams):
    """
    save BPETokenizerParams to output_dir.
    BPETokenizerParams.vocab -> output_dir/{name}-vocab.json
    BPETokenizerParams.merges -> output_dir/{name}-merges.txt
    """
    if(os.path.exists(output_dir) == False): os.mkdir(output_dir)
    name = params.name if params.name != None else "default"

    # to hexadecimal strings of the byte sequences
    vocab_hex = {token_id: token_bytes.hex() for token_id, token_bytes in params.vocab.items()}
    with open(os.path.join(output_dir, f"{name}-vocab.json"), "w", encoding="utf-8") as f_vocab:
        json.dump(vocab_hex, f_vocab, indent=4)
        print(f"vocab saved in {os.path.abspath(os.path.join(output_dir, f"{name}-vocab.json"))}")

    with open(os.path.join(output_dir, f"{name}-merges.txt"), "w", encoding="utf-8") as f_merges:
        for token1, token2 in params.merges:
            f_merges.write(f"{token1.hex()} {token2.hex()}\n")
        print(f"merges saved in {os.path.join(output_dir, f"{name}-merges.txt")}")


def load_bpe_from_dir(output_dir:str, name:str) -> BPETokenizerParams:
    """
    load BPETokenizerParams from output_dir.
    output_dir/{name}-vocab.json -> BPETokenizerParams.vocab
    output_dir/{name}-merges.txt -> BPETokenizerParams.merges

    e.g., load_bpe("data/TinyStoriesV2-GPT4", "TinyStoriesV2-GPT4-valid")
    """
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []

    with open(os.path.join(output_dir, f"{name}-vocab.json"), "r", encoding="utf-8") as f_vocab:
        vocab_hex = json.load(f_vocab)
    vocab = { int(idx): bytes.fromhex(bstring) for idx, bstring in vocab_hex.items() }

    with open(os.path.join(output_dir, f"{name}-merges.txt"), "r", encoding="utf-8") as f_merges:
        for line in f_merges:
            try:
                token1, token2 = bytes.fromhex(line.split(' ')[0]), bytes.fromhex(line.split(' ')[1])
                merges.append((token1, token2))
            except:
                pass

    return BPETokenizerParams(vocab, merges, name)


def load_bpe_from_file(vocab_filepath:str, merges_filepath:str) -> BPETokenizerParams:
    """
    load BPETokenizerParams from output_dir.
    vocab_filepath -> BPETokenizerParams.vocab
    merges_filepath -> BPETokenizerParams.merges
    """
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []
    with open(vocab_filepath, "r", encoding="utf-8") as f_vocab:
        vocab_hex = json.load(f_vocab)
    vocab = { int(idx): bytes.fromhex(bstring) for idx, bstring in vocab_hex.items() }

    with open(merges_filepath, "r", encoding="utf-8") as f_merges:
        for line in f_merges:
            try:
                token1, token2 = bytes.fromhex(line.split(' ')[0]), bytes.fromhex(line.split(' ')[1])
                merges.append((token1, token2))
            except:
                pass
    return BPETokenizerParams(vocab, merges)

# example BPETokenizer in lecture_01.py
class BPETokenizer_example(Tokenizer):
    """BPE tokenizer given a set of merges and a vocabulary."""
    def __init__(self, params: BPETokenizerParams):
        self.params = params
    def encode(self, string: str) -> list[int]:
        indices = list(map(int, string.encode("utf-8")))  # @inspect indices
        # Note: this is a very slow implementation
        for pair, new_index in self.params.merges.items():  # @inspect pair, @inspect new_index
            indices = merge(indices, pair, new_index)
        return indices
    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.params.vocab.get, indices))  # @inspect bytes_list
        string = b"".join(bytes_list).decode("utf-8")  # @inspect string
        return string


# slow version in lecture_01.py
def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:  # @inspect indices, @inspect pair, @inspect new_index
    """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
    new_indices = []  # @inspect new_indices
    i = 0  # @inspect i
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices

def test_encode():
    from tests.test_tokenizer import test_tinystories_sample_roundtrip
    import time
    start = time.time()
    test_tinystories_sample_roundtrip()
    end = time.time()
    print(f"encoding time (seconds): {end - start}")

if __name__ == "__main__":
    test_encode()