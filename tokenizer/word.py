import logging
import re

from datasets.arrow_dataset import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class WordTokenizer:
    def __init__(self, vocab_size: int, unknown_token: str = None):
        if not unknown_token:
            unknown_token = "<UNK>"
        self.unknown_token = unknown_token
        self.word_to_id = {}
        self.id_to_word = {}
        self.word_to_id[self.unknown_token] = 0
        self.id_to_word[0] = self.unknown_token
        self.next_id = 1  # Reserve 0 for UNKNOWN words
        self._n_vocab = vocab_size
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

    def build_vocab(self, text):
        words = WordTokenizer.extract_words(text)
        for word in tqdm(words, desc="Building vocabulary:"):
            if word not in self.word_to_id and len(self.word_to_id) < self._n_vocab:
                self.word_to_id[word] = self.next_id
                self.id_to_word[self.next_id] = word
                self.next_id += 1

    def encode(
        self, src_txt: str, tgt_txt: str
    ) -> tuple[list[int], list[int]] | tuple[None, None]:
        pass
        # self.build_vocab(text)
        # if len(self.word_to_id) < 2 or len(self.id_to_word) < 2:
        #     logger.error("Please build the vocab first")
        #     return None
        # words = WordTokenizer.extract_words(text)
        # return [self.word_to_id.get(word, self.word_to_id[self.unknown_token]) for word in words]

    def decode(self, ids):
        if len(self.word_to_id) < 2 or len(self.id_to_word) < 2:
            logger.error("Please build the vocab first")
            return None
        return " ".join(self.id_to_word[i] for i in ids)

    @staticmethod
    def extract_words(text):
        return re.findall(r"\b\w+\b", text.lower())

    @property
    def n_vocab(self):
        return self._n_vocab

    def train(self, dataset: Dataset, lang_keys: list = None):
        pass
