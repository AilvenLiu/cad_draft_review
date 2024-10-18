import collections

class Vocab:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<UNK>": 1}
        self.freq_threshold = freq_threshold

    def build_vocabulary(self, sentence_list):
        frequencies = collections.defaultdict(int)
        idx = 2

        for sentence in sentence_list:
            frequencies[sentence] += 1

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokenized_text = text.split()
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]