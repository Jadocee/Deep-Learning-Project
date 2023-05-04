import string
from itertools import chain
from typing import List, Set

from datasets import Dataset
from nltk import WordNetLemmatizer, word_tokenize, PorterStemmer
from nltk.corpus import stopwords
from nltk.lm import Vocabulary
from torchtext.vocab import build_vocab_from_iterator, Vocab

STOP_WORDS: Set[str] = set(stopwords.words("english"))
"""
A set of stop words from the nltk library.
"""


class DataProcessingUtils:

    @staticmethod
    def standardise_text(text: str, max_tokens: int, lemmatise: bool = True, stem: bool = False) -> List[str]:
        # Convert to lower case
        text = text.lower()

        # Remove punctuation
        text = "".join([char for char in text if char not in string.punctuation])

        # Tokenise
        tokens: List[str] = word_tokenize(text=text, language="english")

        # Remove stop words
        tokens = [token for token in tokens if token not in STOP_WORDS]

        # Finalise
        if lemmatise and stem:
            raise Exception("Cannot lemmatise and stem at the same time")
        elif lemmatise:
            tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens]
        elif stem:
            tokens = [PorterStemmer().stem(token) for token in tokens]

        return tokens[:max_tokens]

    @staticmethod
    def create_vocab(train_data: Dataset, valid_data: Dataset, test_data: Dataset) -> Vocab:
        vocab: Vocab = build_vocab_from_iterator(
            chain(train_data, valid_data, test_data),
            specials=["<unk>", "<pad>"],
            min_freq=5
        )
        vocab.set_default_index(vocab["<unk>"])
        return vocab

    @staticmethod
    def create_vocab_1(word_tokens: List[str]) -> Vocabulary:
        vocab: Vocabulary = Vocabulary(counts=word_tokens, unk_cutoff=5)
        return vocab
