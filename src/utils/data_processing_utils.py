import string
import warnings
from collections import Counter
from typing import List, Set, Iterable, Union

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.lm import Vocabulary
from nltk.stem import WordNetLemmatizer, PorterStemmer
from torchtext.vocab import build_vocab_from_iterator, Vocab


class DataProcessingUtils:
    STOP_WORDS: Set[str] = set(stopwords.words("english"))
    """
    A set of stop words from the nltk library.
    """

    @staticmethod
    def standardise_text(text: str, max_tokens: int, lemmatise: bool = True, stem: bool = False) -> List[str]:
        # Convert to lower case
        text = text.lower()

        # Remove punctuation
        text = "".join([char for char in text if char not in string.punctuation])

        # Tokenise
        tokens: List[str] = word_tokenize(text=text, language="english")

        # Remove stop words
        tokens = [token for token in tokens if token not in DataProcessingUtils.STOP_WORDS]

        # Finalise
        if lemmatise and stem:
            raise Exception("Cannot lemmatise and stem at the same time")
        elif lemmatise:
            tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens]
        elif stem:
            tokens = [PorterStemmer().stem(token) for token in tokens]

        return tokens[:max_tokens]

    @staticmethod
    def create_vocab(tokens: Iterable) -> Vocab:
        vocab: Vocab = build_vocab_from_iterator(
            tokens,
            specials=["<unk>", "<pad>"],
            min_freq=5
        )
        vocab.set_default_index(vocab["<unk>"])
        return vocab

    @staticmethod
    def create_vocab_1(word_tokens: Iterable) -> Vocabulary:
        warnings.warn("This method is deprecated. Use create_vocab instead.", DeprecationWarning)
        word_counts = Counter(word_tokens)
        vocab: Vocabulary = Vocabulary(counts=word_counts, unk_cutoff=5)
        return vocab

    @staticmethod
    def vocabularise_text(tokens: List[str], vocab: Union[Vocab, Vocabulary]) -> List[int]:
        ids = [vocab[token] for token in tokens]
        return ids
