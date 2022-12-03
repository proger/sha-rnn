import os
import torch

from tqdm import tqdm
from collections import Counter, defaultdict


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = defaultdict(int)
        self.total = 0

    def add_word(self, word, freq=1):
        try:
            token_id = self.word2idx[word]
        except KeyError:
            self.idx2word.append(word)
            token_id = self.word2idx[word] = len(self.idx2word) - 1
        if freq:
            self.counter[token_id] += freq
            self.total += freq
        return token_id

    def update(self, words: Counter):
        for word in words:
            self.add_word(word, freq=0)
        self.counter.update(words)
        self.total += words.total()

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        for byte in range(256):
            self.dictionary.add_word(byte.to_bytes(1, byteorder='little'), freq=0)
        self.train = self.tokenize(os.path.join(path, 'train.txt.raw'), construct_dictionary=True)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt.raw'))
        self.test = self.tokenize(os.path.join(path, 'test.txt.raw'))

    def tokenize(self, path, construct_dictionary=False):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        return torch.ByteTensor(torch.ByteStorage.from_file(path, shared=True, size=os.stat(path).st_size))
