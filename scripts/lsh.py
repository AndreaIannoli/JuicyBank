import hashlib
import random
from typing import List, Set

# Constants for hash functions
MAX_HASH = (1 << 32) - 1  # Maximum hash value (32-bit)
HASH_PRIME = 4294967311    # A large prime number near 2^32


def shingle(string: str, k: int) -> Set[str]:
    """Convert a string into k-shingles (n-grams)."""
    return {string[i:i + k] for i in range(len(string) - k + 1)} if len(string) >= k else {string}


def generate_hash_functions(num_hashes: int):
    """Generate hash function parameters."""
    random.seed(42)  # For reproducibility
    a_list = random.sample(range(1, HASH_PRIME), num_hashes)
    b_list = random.sample(range(0, HASH_PRIME), num_hashes)
    return a_list, b_list


def minhash(signature_set: Set[str], num_hashes: int, a_list: List[int], b_list: List[int]) -> List[int]:
    """Generate MinHash signature for a set of shingles."""
    signature = [MAX_HASH] * num_hashes

    for shingle_str in signature_set:
        # Convert shingle to an integer hash value
        shingle_int = int(hashlib.md5(shingle_str.encode(
            'utf-8')).hexdigest(), 16) % HASH_PRIME
        for i in range(num_hashes):
            a = a_list[i]
            b = b_list[i]
            hash_value = (a * shingle_int + b) % HASH_PRIME
            if hash_value < signature[i]:
                signature[i] = hash_value
    return signature


def lsh(strings: List[str]) -> Set[frozenset]:
    """
    Apply LSH to a set of strings and return sets of words sharing at least one bucket.

    Args:
        strings: List of strings to process.

    Returns:
        A set of frozensets, each containing words that share at least one bucket.
    """
    k = 1  # Length of shingles (n-grams)
    num_hashes = 100  # Number of hash functions
    bands = 12  # Number of bands

    # Generate hash function parameters
    a_list, b_list = generate_hash_functions(num_hashes)

    # Generate MinHash signatures for each string
    signatures = {s: minhash(shingle(s, k), num_hashes,
                             a_list, b_list) for s in strings}
    bucket_map = {}

    # Assign words to buckets
    for s, sig in signatures.items():
        for b in range(bands):
            # Divide the signature into bands
            rows_per_band = num_hashes // bands
            start = b * rows_per_band
            end = start + rows_per_band
            band = tuple(sig[start:end])
            # Hash the band to get a bucket ID
            bucket_id = hashlib.md5(str(band).encode()).hexdigest()
            if bucket_id not in bucket_map:
                bucket_map[bucket_id] = set()
            bucket_map[bucket_id].add(s)

    # Collect sets of words that share buckets
    from collections import defaultdict

    word_groups = []
    bucket_words = list(bucket_map.values())

    # Union-Find data structure to merge groups
    parent = {}

    def find(word):
        if parent[word] != word:
            parent[word] = find(parent[word])
        return parent[word]

    def union(word1, word2):
        root1 = find(word1)
        root2 = find(word2)
        if root1 != root2:
            parent[root2] = root1

    # Initialize parent pointers
    all_words = set(strings)
    for word in all_words:
        parent[word] = word

    # Union words that appear in the same bucket
    for words_in_bucket in bucket_words:
        words_list = list(words_in_bucket)
        for i in range(len(words_list) - 1):
            union(words_list[i], words_list[i + 1])

    # Group words by their root parent
    groups = defaultdict(set)
    for word in all_words:
        root = find(word)
        groups[root].add(word)

    # Convert groups to a set of frozensets
    final_groups = set(frozenset(group) for group in groups.values())

    return final_groups
