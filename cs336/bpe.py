import regex as re

text = "some text that i'll pre-tokenize"
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
parts = re.findall(PAT, text)
print(parts)


def read_from_file(input_path: str) -> str:
    with open(input_path, "r") as f:
        text = f.read()
    return text


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    # input:             "low low low lower<endoftext>lowerest"
    # special_tokens:    "<endoftext>"
    # vocab_size:        260

    # 1. create the token table
    # e.g. 1 - 255 -> byte
    # 256 -> b'<endoftext>'
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])

    # 2. cut the words by special tokens
    # "low low low lower<endoftext>lowerest"
    # chunks: ["low low low lower", "lowerest"]
    raw = read_from_file(input_path)
    split_pat = "|".join(re.escape(t) for t in special_tokens)
    chunks = re.split(split_pat, raw)

    # 3. tokenized
    # counter:
    # { "low": 1, " low": 2, "lower": 1, "lowerest": 1}
    counts = {}
    for chunk in chunks:
        for m in re.finditer(PAT, chunk):
            counts[m.group().encode("utf-8")] += 1

    # 4. to single bytes -> frequency
    # { "l o w": 1, " l o w": 2, "l o w e r": 1, "l o w e r e s t": 1}
    byte_freqs = []
    byte_words = []

    for k, v in counts:
        byte_words.add([bytes([c]) for c in k])
        byte_freqs.add(v)

    # 5. merge adjacent bytes
    # (l, o)  = 1 + 2 + 1 + 1 = 5
    # (o, w)  = 5
    # (␣, l)  = 2 + 1 = 3
    # (w, e)  = 2
    # (e, r)、(e, s)、(s, t) = 1

    merged_count = {}
    for byte_word in byte_words:
        for i in range(len(byte_word)):
            j = i + 1
            if j >= len(byte_word):
                break
            merged_count

    pass
