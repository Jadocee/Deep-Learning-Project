import string
import warnings
from collections import Counter
from typing import List, Set, Iterable, Union

from contractions import fix as contractions_fix
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
    def spacy_tokeniser(text: str) -> List[str]:
        # TODO: Implement me!
        raise NotImplementedError
        # tokens: List[str] = parser(text)

    @staticmethod
    def standardise_text(text: str, max_tokens: int, lemmatise: bool = True, stem: bool = False) -> List[str]:
        """
        Standardises the given text by converting to lower case, expanding contractions, removing punctuation,
        tokenising, and removing stop words. It then optionally performs lemmatisation or stemming.

        Args:
            text (str): The text to be standardised.
            max_tokens (int): Maximum number of tokens to be returned after text standardisation.
            lemmatise (bool, optional): If True, performs lemmatisation on the tokens. Defaults to True.
            stem (bool, optional): If True, performs stemming on the tokens. Defaults to False.

        Raises:
            Exception: If both lemmatise and stem are set to True.

        Returns:
            List[str]: List of standardised tokens.
        """
        # Convert to lower case
        text = text.lower()
        # Expand contractions
        contractions_fix(text)
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
    def create_vocab(tokens: Iterable, min_freq: int = 1) -> Vocab:
        """
        Builds a vocabulary from an iterable of tokens. This method uses a minimum frequency to filter out rare words
        and includes special tokens for unknown and padding cases. The default index is set to the unknown token.

        Args:
            tokens (Iterable): An iterable of tokens to build the vocabulary from.
            min_freq (int, optional): The minimum frequency for a token to be included in the vocabulary. Defaults to 1.

        Returns:
            Vocab: The created vocabulary.
        """
        vocab: Vocab = build_vocab_from_iterator(
            tokens,
            specials=["<unk>", "<pad>"],
            min_freq=min_freq
        )
        vocab.set_default_index(vocab["<unk>"])
        return vocab

    @staticmethod
    def create_vocab_1(word_tokens: Iterable) -> Vocabulary:
        """
        Deprecated method for creating a vocabulary from an iterable of word tokens. Use create_vocab instead.

        Args:
            word_tokens (Iterable): An iterable of word tokens to build the vocabulary from.

        Returns:
            Vocabulary: The created vocabulary.
        """
        warnings.warn("This method is deprecated. Use create_vocab instead.", DeprecationWarning)
        word_counts = Counter(word_tokens)
        vocab: Vocabulary = Vocabulary(counts=word_counts, unk_cutoff=5)
        return vocab

    @staticmethod
    def vocabularise_text(tokens: List[str], vocab: Union[Vocab, Vocabulary]) -> List[int]:
        """
        Converts a list of tokens into their corresponding numerical identifiers using the provided vocabulary.

        Args:
            tokens (List[str]): A list of text tokens to be converted.
            vocab (Union[Vocab, Vocabulary]): The vocabulary to use for the conversion. Can be either a Vocab or Vocabulary object.

        Returns:
            List[int]: A list of numerical identifiers corresponding to the input tokens.
        """
        ids = [vocab[token] for token in tokens]
        return ids
