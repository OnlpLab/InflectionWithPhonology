from collections import Counter

class Vocab(object):
    #  A generic class for representing a vocabulary of some language, be it letters, features, actions or words
    def __init__(self, w2i=None, encoding='utf8'):
        if w2i is None:
            self.w2i = dict()
        else:
            self.w2i = dict(w2i)

        self.i2w = {i: w for w, i in self.w2i.items()}
        self.encoding = encoding
        self.freqs = Counter(list(self.i2w.keys()))

    @classmethod
    def from_list(cls, words, encoding='utf8'):
        w2i, idx = {}, 0
        for word in set(words):
            if encoding:
                word = word.decode(encoding)
            w2i[word] = idx
            idx += 1

        return Vocab(w2i, encoding=encoding)

    def __getitem__(self, word) -> int:
        # encodes the word if it is not in vocab
        if self.encoding:
            word = word.decode(self.encoding)

        if word in self.w2i:
            idx = self.w2i[word]
        else:
            idx = self.size()
            self.w2i[word] = idx
            self.i2w[idx] = word

        self.freqs[idx] += 1

        return idx

    def __contains__(self, word):
        if self.encoding:
            word = word.decode(self.encoding)
        return word in self.w2i

    def keys(self):
        return list(self.w2i.keys())

    def freq(self):
        return dict(self.freqs)

    def __repr__(self):
        return str(self.w2i)

    def __len__(self):
        return self.size()

    def size(self):
        return len(list(self.w2i.keys()))

    def printer(self):
        print("\nPrinting object info:")
        print(f"self.encoding = {self.encoding}")

        print(f"self.w2i = {self.w2i}")
        print(f"self.i2w = {self.i2w}")

        print(f"self.freqs = {self.freqs}")

        print(f"self.keys() = {self.keys()}")
        print(f"self.freq() = {self.freq()}")
        print(f"self.size() = {self.size()}\n")