import os
import regex as re
from typing import BinaryIO
from collections import defaultdict
from cs336_basics.bpe_tokenizer import BPETokenizerParams, merge, load_bpe_from_dir, save_bpe
from concurrent.futures import ProcessPoolExecutor, as_completed

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def remove_special_tokens(string: str, special_tokens: list[str] = None) -> list[str]:
    """
    Removing special tokens in a given string before pre-tokenization.
    Return a list of splitted strings
    """
    if("" in special_tokens): special_tokens.remove("")
    if(special_tokens != None):
        pattern = '|'.join(re.escape(token) for token in special_tokens)
        split_strings = re.split(pattern, string)
    else:
        split_strings = [string]
    return split_strings


def pre_tokenization_chunk(chunk_string: str, special_tokens: list[str] = None) -> dict[tuple[bytes], int]:
    split_chunks = remove_special_tokens(chunk_string, special_tokens) # remove special tokens in chunk_string

    res = defaultdict(int)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    compiled_pat = re.compile(PAT)
    for chunk in split_chunks:
        # a regex-based pre-tokenizer (used by GPT-2)
        matches = compiled_pat.finditer(chunk)
        for match in matches:
            token_string = match.group(0)
            res[ token_string ] += 1
    return res


def pre_tokenization_corpus(corpus_path: str, special_tokens: list[str] = None, num_processes: int = 4) -> dict[tuple[bytes], int]:
    num_processes = max(1, num_processes)
    
    ## reading
    print(f"reading {os.path.abspath(corpus_path)} ...")
    with open(corpus_path, "rb") as file:
        # get string
        chunks = []
        boundaries = find_chunk_boundaries(file, num_processes, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            file.seek(start)
            chunk_string: str = file.read(end - start).decode("utf-8", errors="replace")
            chunks.append(chunk_string)
        
    # parallelism for pre-tokenization
    res_chunks = []
    print(f"start pre-tokenization (chunks = {len(chunks)}, workers = {num_processes})...")
    with ProcessPoolExecutor(max_workers= min(num_processes, os.cpu_count()) ) as executor:
        futures = []
        for chunk_string in chunks:
            future = executor.submit(pre_tokenization_chunk, chunk_string, special_tokens)
            futures.append(future)
        
        #for future in tqdm(as_completed(futures), total=len(futures), leave=False):
        for future in as_completed(futures):
            res_chunk = future.result()
            res_chunks.append(res_chunk)

    res_corpus = defaultdict(int)
    for res_chunk in res_chunks:
        for key, value in res_chunk.items():
            indices = list(map(int, key.encode("utf-8")))
            res_corpus[tuple(indices)] += value

    return res_corpus


# my implementation
def train_bpe(input_file: str, vocab_size: int, special_tokens: list[str]) -> BPETokenizerParams:
    """
    Problem (train_bpe): BPE Tokenizer Training
    Write a function that, given a path to an input text file, trains a (byte-level) BPE tokenizer.
    vocab: dict[int, bytes]    # index -> bytes
    merges: list[tuple[bytes, bytes]]  # bytes (<token1>, <token2>)
    """
    frequency_table = pre_tokenization_corpus(input_file, special_tokens, os.cpu_count()) # dict[tuple[bytes], int]
    # init
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []
    vocab_idx = 0
    for x in range(256):
        vocab[vocab_idx] = bytes([x])
        vocab_idx+=1
    for token in special_tokens:
        vocab[vocab_idx] = token.encode("utf-8")
        vocab_idx+=1

    # count the number of occurrences of each pair of tokens
    # as a cache for speedup
    print("start training...")
    counts = defaultdict(int)
    for indices, frequency in frequency_table.items():
        for index1, index2 in zip(indices, indices[1:]):  # For each adjacent pair
            counts[(index1, index2)] += frequency  # @inspect counts
    
    while(vocab_idx < vocab_size):
        #Find the most common pair.
        #pair = max(counts, key=counts.get)  # @inspect pair
        pair = max(
            counts.items(),
            key=lambda x: (
                x[1],  
                vocab[x[0][0]].decode("utf-8", errors="replace"),
                vocab[x[0][1]].decode("utf-8", errors="replace")
            )
        )[0]
        if(counts[pair] == 1): break
        index1, index2 = pair
        counts.pop(pair) # update counts
        
        # Merge that pair.
        # merges[pair] = new_index  # @inspect merges
        vocab[vocab_idx] = vocab[index1] + vocab[index2]  # @inspect vocab
        merges.append(( vocab[index1], vocab[index2] ))
        for key in list(frequency_table.keys()):
            new_key = []
            ii, flag = 0, 0
            while ii < len(key):
                if ii + 1 < len(key) and key[ii] == pair[0] and key[ii + 1] == pair[1]:
                    new_key.append(vocab_idx) # vocab_idx is the new index
                    ii += 2
                    flag = 1 # merged
                else:
                    new_key.append(key[ii])
                    ii += 1
            
            # update count
            if(flag == 0): continue
            frequency = frequency_table[key]
            old_count, new_count = defaultdict(int), defaultdict(int)
            old_idx, new_idx = zip(key, key[1:]), zip(new_key, new_key[1:])
            for old_pair in old_idx: old_count[old_pair]+=1
            for new_pair in new_idx: new_count[new_pair]+=1
            for tmp in set(old_count.keys()).union(new_count.keys()):
                if(tmp!=pair): counts[ tmp ] += (new_count[tmp] - old_count[tmp]) * frequency
            
            frequency_table[tuple(new_key)] = frequency_table.pop(key)
        
        vocab_idx += 1
    
    return BPETokenizerParams(vocab, merges, "".join(os.path.basename(input_file).split('.')[:-1]))


def train_bpe_tinystories():
    """
    Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size
    of 10,000. Make sure to add the TinyStories <|endoftext|> special token to the vocabulary.
    Serialize the resulting vocabulary and merges to disk for further inspection. How many hours
    and memory did training take? What is the longest token in the vocabulary? Does it make sense?

    training set: data/TinyStoriesV2-GPT4-train.txt
    """
    import time
    dir_path = os.path.dirname( os.path.abspath(__file__) )

    time_start = time.time()
    # for test (around 6 seconds in my machine)
    tinystories_res = train_bpe(os.path.join(dir_path, "../", "data", "TinyStoriesV2-GPT4-valid.txt"), 1000, special_tokens=["<|endoftext|>"])
    # training task in the assignment 1
    # tinystories_res = train_bpe(os.path.join(dir_path, "../", "data", "TinyStoriesV2-GPT4-train.txt"), 10000, special_tokens=["<|endoftext|>"])
    time_end = time.time()
    print(f"train_bpe time (seconds): {time_end - time_start}")

    # to disk
    output_dir = os.path.join(dir_path, "../", "data", "TinyStoriesV2-GPT4")
    save_bpe(output_dir, tinystories_res)


def train_bpe_expts_owt():
    """
    Train a byte-level BPE tokenizer on the OpenWebText dataset, using a maximum vocabulary
    size of 32,000. Serialize the resulting vocabulary and merges to disk for further inspection. What
    is the longest token in the vocabulary? Does it make sense?

    training set: data/owt_train.txt
    """
    import time
    dir_path = os.path.dirname( os.path.abspath(__file__) )

    time_start = time.time()
    # for test (around 320 seconds in my machine)
    tinystories_res = train_bpe(os.path.join(dir_path, "../", "data", "owt_valid.txt"), 1000, special_tokens=["<|endoftext|>"])
    # training task in the assignment 1
    # tinystories_res = train_bpe(os.path.join(dir_path, "../", "data", "owt_train.txt"), 32000, special_tokens=["<|endoftext|>"])
    time_end = time.time()
    print(f"training time (seconds): {time_end - time_start}")

    # to disk
    output_dir = os.path.join(dir_path, "../", "data", "owt")
    save_bpe(output_dir, tinystories_res)


# the slow version in lecture_01.py
# def train_bpe_example(string: str, num_merges: int):  # @inspect string, @inspect num_merges
def train_bpe_example(input_path: str, vocab_size: int, special_tokens: list[str]) -> BPETokenizerParams:
    #Start with the list of bytes of string.
    with open(input_path, "r", encoding="utf-8") as file:
        string = file.read()
    indices = list(map(int, string.encode("utf-8")))  # @inspect indices
    #merges: dict[tuple[int, int], int] = {}  # index1, index2 => merged index
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # index -> bytes
    merges: list[tuple[bytes, bytes]] = []

    i = 256
    while(i < vocab_size):
        #Count the number of occurrences of each pair of tokens
        counts = defaultdict(int)
        for index1, index2 in zip(indices, indices[1:]):  # For each adjacent pair
            counts[(index1, index2)] += 1  # @inspect counts
        
        #Find the most common pair.
        pair = max(counts, key=counts.get)  # @inspect pair
        index1, index2 = pair
        
        # Merge that pair.
        new_index = i  # @inspect new_index
        i += 1
        vocab[new_index] = vocab[index1] + vocab[index2]  # @inspect vocab
        merges.append(( vocab[index1], vocab[index2] ))
        indices = merge(indices, pair, new_index)  # @inspect indices
    return BPETokenizerParams(vocab, merges)


def my_test_train_bpe():
    # testing train_bpe use `tests/fixtures/corpus.en`
    dir_path = os.path.dirname( os.path.abspath(__file__) )
    return train_bpe(os.path.join(dir_path, "../", "tests/fixtures", "corpus.en"), 500, special_tokens=["<|endoftext|>"])


def my_test_save_and_load():
    import tempfile
    with tempfile.TemporaryDirectory(dir='./') as temp_dir:
        paras_trained = my_test_train_bpe()
        save_bpe(temp_dir, paras_trained)
        paras_loaded = load_bpe_from_dir(temp_dir, paras_trained.name)
        assert paras_trained.merges == paras_loaded.merges
        assert set(paras_trained.vocab.keys()) == set(paras_loaded.vocab.keys())
        assert set(paras_trained.vocab.values()) == set(paras_loaded.vocab.values())


if(__name__ == "__main__"):
    # my_test_train_bpe()
    # train_bpe_tinystories()
    # train_bpe_expts_owt()
    my_test_save_and_load()