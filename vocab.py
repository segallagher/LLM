import pandas as pd
import json
from tqdm import tqdm
from collections import defaultdict
import string

## Byte Pair encoding utilities
# setences to character-level vocabulary
def initialize_vocab(sentences: list[str]) -> dict[str, int]:
    vocab: defaultdict[str, int] = defaultdict(int)
    for sentence in sentences:
        for word in sentence.strip().split():
            chars = list(word) + ['</w>']
            key = ' '.join(chars)
            vocab[key] += 1
    return dict(vocab)

# count symbol pairs
def get_stats(vocab: dict[str, int]) -> dict[tuple, int]:
    # loop over each word in vocab and their frequencies
    pairs: defaultdict[tuple, int] = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        # for each pair in each word, add the frequency of the word (since that is the same frequncy as that pair from that word)
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] +=  freq
    # return all pairs of all letters in vocabulary
    return dict(pairs)

def merge_vocab(pair: tuple, vocab: dict[str, int]) -> dict[str, int]:
    new_vocab = {}
    # find string to replace like "a b"
    bigram = ' '.join(pair)
    # replace with "ab"
    replacement = ''.join(pair)
    # for word in vocab, if can find pair as bigram, replace with replacement
    for word in vocab:
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = vocab[word]
    # return vocab where all instances of pair as bigram are merged
    return new_vocab

# returns true if pair is valid, otherwise false
def is_valid_pair(pair: tuple[str, str]) -> bool:
    # tokens containing invalid_tokens are not eligable for merging
    invalid_tokens = ['</w>'] + list(string.punctuation)
    a, b = pair
    for token in invalid_tokens:
        if token in a or token in b:
            return False
    return True

# Build a vocabulary using byte-pair encoding
def build_vocab(data: list[str], target_vocab_size:int=1000):
    # Initialize vocabulary
    vocab=initialize_vocab(data)
    # Apply byte-pair encoding
    actual_vocab_size = 0
    pbar = tqdm(total=target_vocab_size, desc="Building vocab")
    while actual_vocab_size < target_vocab_size:
        # Get all pairs 
        pairs: dict [tuple, int] = get_stats(vocab)

        # THIS COULD/SHOULD BE OPTIMIZED
        # KNOWN BUG IF VOCAB IS TOO HIGH FILTERED_PAIRS MIGHT BE EMPTY
        # Filter pairs
        filtered_pairs = {key: value for key, value in pairs.items() if is_valid_pair(key)}

        # Get most frequent pair
        best = max(filtered_pairs, key=filtered_pairs.get)
        # merge most frequent pair
        vocab = merge_vocab(best, vocab)
        # update progress bar
        actual_vocab_size = len(find_unique_tokens(vocab))
        pbar.n = actual_vocab_size
        pbar.refresh()
    pbar.close
    return vocab

# Takes a list of the vocabulary and returns a set of all unique tokens
def find_unique_tokens(vocab: list[str]):
    tokens=set()
    for word in vocab:
        tokens.update(word.split())
    return tokens

# given a set of tokens, create a dict mapping tokens to integers with additional utility tokens
def create_token_id_map(token_set:set):
    token_ids = {"<unk>" : 0, "<pad>": 1}
    for idx, token in enumerate(sorted(token_set), start=len(token_ids)):
        token_ids[token] = idx
    return token_ids

# Create vocab given a dataset which is a DataFrame or a pytorch SubSet, a target vocab size, and a path to output
def create_vocab(dataset: pd.DataFrame, target_vocab_size: int, vocab_output_path: str) -> None:

    merged_dataset_list = dataset["input"].to_list() + dataset["output"].to_list()

    # Build Vocabulary
    vocab = build_vocab(data=merged_dataset_list, target_vocab_size=target_vocab_size)

    token_set = find_unique_tokens(vocab)

    vocab = create_token_id_map(token_set)

    with open(vocab_output_path, 'w') as f:
        json.dump(vocab, f)