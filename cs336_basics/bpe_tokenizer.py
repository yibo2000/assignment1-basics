# from lecture_01.py
# export PYTHONPATH=./

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
    
    def __init__(self, _vocab, _merges):
        self.vocab = _vocab
        self.merges = _merges


class BPETokenizer(Tokenizer):
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


# slow version  
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