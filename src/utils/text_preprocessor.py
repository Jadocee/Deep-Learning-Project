import string
from itertools import chain
from os.path import join
from pathlib import Path
from typing import Set, Optional, List, Iterable

import contractions
import torch
from datasets import DatasetDict, Dataset
from nltk import word_tokenize, WordNetLemmatizer, PorterStemmer, download
from nltk.corpus import stopwords
from numpy import zeros, ndarray, vstack
from torchtext.vocab import Vocab, build_vocab_from_iterator

from utils.definitions import VOCABS_DIR


class TextPreprocessor:
    """
    Class for handling the pre-processing of text.

    This class is used to perform the various text pre-processing techniques used in this project, such as
    standardisation and vectorisation. It also contains methods for saving and loading vocabularies.

    Attributes:
        __stop_words (Set[str]): The set of stop words to be used for text pre-processing.
        __vocab (Optional[Vocab]): The vocabulary created from the training and validation data, used for vectorisation.
        __oov_token (str): The token to be used for out-of-vocabulary words.
        __pad_token (str): The token to be used for padding.
        __max_tokens (int): The maximum number of tokens to be used for vectorisation.
        __normalisation_method (str): The normalisation method to be used for text pre-processing.
        __encoding_method (str): The encoding method to be used for vectorisation.
        __remove_stop_words (bool): Whether to remove stop words during text pre-processing.
        __lower_case (bool): Whether to convert all text to lower case during text pre-processing.
        __expand_contractions (bool): Whether to expand contractions during text pre-processing.
        __remove_punctuation (bool): Whether to remove punctuation during text pre-processing.
        __perform_normalisation (bool): Whether to perform normalisation during text pre-processing.
        __perform_vocabularisation (bool): Whether to perform vocabularisation during text pre-processing.
        __encode (bool): Whether to encode the text during text pre-processing.
    """

    __stop_words: Set[str]
    __vocab: Optional[Vocab]
    __oov_token: str
    __pad_token: str
    __max_tokens: int
    __normalisation_method: str
    __encoding_method: str

    __remove_stop_words: bool
    __lower_case: bool
    __expand_contractions: bool
    __remove_punctuation: bool
    __perform_normalisation: bool
    __perform_vocabularisation: bool
    __encode: bool

    def __init__(self,
                 remove_stop_words: bool = True,
                 lower_case: bool = True,
                 expand_contractions: bool = True,
                 remove_punctuation: bool = True,
                 normalise: bool = True,
                 encode: bool = False,
                 stop_words: Optional[Set[str]] = None,
                 normalisation_method: str = "lemmatise",
                 encoding_method: str = "multi-hot",
                 max_tokens: int = 1000,
                 oov_token: str = "<unk>",
                 pad_token: str = "<pad>"
                 ) -> None:
        """
        Initialises the TextPreprocessor class.

        Args:
            remove_stop_words (bool, optional): Whether to remove stop words during the standardisation process.
             Defaults to True.
            lower_case (bool, optional): Whether to convert the text to lower case during the standardisation process.
                Defaults to True.
            expand_contractions (bool, optional): Whether to expand contractions during the standardisation process.
                Defaults to True.
            remove_punctuation (bool, optional): Whether to remove punctuation during the standardisation process.
                Defaults to True.
            normalise (bool, optional): Whether to normalise the text during the standardisation process. Defaults to
                True.
            encode (bool, optional): Whether to encode the text during the standardisation process. Defaults to False.
            stop_words (Optional[Set[str]], optional): A set of stop words to use during the standardisation process. If
                None, the stop words from the nltk library will be used. Defaults to None.
            normalisation_method (str, optional): The normalisation method to use during the standardisation process.
                Must be either "lemmatise" or "stem". Defaults to "lemmatise".
            encoding_method (str, optional): The encoding method to use during the standardisation process. Must be
                either "multi-hot", "tf-idf" or "one-hot". Defaults to "multi-hot".
            max_tokens (int, optional): The maximum number of tokens to use during the standardisation process. The list
                of tokens will be truncated to this length. Defaults to 1000.
            oov_token (str, optional): The out-of-vocabulary token to use during the standardisation process. Defaults
                to "<unk>".
            pad_token (str, optional): The padding token to use during the standardisation process. Defaults to "<pad>".

        Raises:
            ValueError: If the normalisation method is not "lemmatise" or "stem".
            ValueError: If the encoding method is not "multi-hot", "tf-idf" or "one-hot".

        Returns:
            None

        Notes:
            - It is recommended to use the default values for the OOV (out-of-vocabulary) and PAD (padding) tokens.
            - Lemmatisation is the process of grouping together the inflected forms of a word, so they can be analysed
                as a single item, identified by the word's lemma, or dictionary form.
            - Stemming is the process of reducing inflected (or sometimes derived) words to their word stem, base or
                root form.
            - It is recommended to use lemmatisation over stemming as it produces more accurate results in most cases.
        """

        if normalisation_method not in ["lemmatise", "stem"]:
            raise ValueError("The normalisation method must be either 'lemmatise' or 'stem'.")

        if encoding_method not in ["multi-hot", "tf-idf", "one-hot"]:
            raise ValueError("The encoding method must be either 'multi-hot', 'tf-idf' or 'one-hot'.")

        download("stopwords", quiet=True)
        download("wordnet", quiet=True)
        download("punkt", quiet=True)

        if stop_words is None:
            self.__stop_words = set(stopwords.words("english"))
        else:
            self.__stop_words = stop_words

        self.__remove_stop_words = remove_stop_words
        self.__lower_case = lower_case
        self.__expand_contractions = expand_contractions
        self.__remove_punctuation = remove_punctuation
        self.__normalisation_method = normalisation_method
        self.__perform_normalisation = normalise
        self.__max_tokens = max_tokens
        self.__oov_token = oov_token
        self.__pad_token = pad_token
        self.__encode = encode
        self.__encoding_method = encoding_method
        self.__vocab = None

    def create_vocab(self, tokens: Iterable, min_freq: int = 1) -> None:
        """
        Builds a vocabulary from an iterable of tokens.

        This method uses a minimum frequency to filter out rare words and includes special tokens for unknown and
        padding cases.

        Args:
            tokens (List[str]): The list of tokens to build the vocabulary from.
            min_freq (int, optional): The minimum frequency for a token to be included in the vocabulary. Defaults to 1.

        Returns:
            None

        Note:
            The default index is set to the unknown token.
        """
        vocab: Vocab = build_vocab_from_iterator(
            tokens,
            specials=[self.__oov_token, self.__pad_token],
            min_freq=min_freq,
        )
        vocab.set_default_index(vocab["<unk>"])
        self.__vocab = vocab

    def save_vocab(self, vocab_name: str) -> None:
        """
        Saves the vocabulary to a file.

        Args:
            vocab_name (str): The name of the vocabulary file to save.

        Returns:
            None
        """
        output_dir: Path = Path(VOCABS_DIR)
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.__vocab, join(VOCABS_DIR, f"{vocab_name}.pth"))

    def load_vocab(self, vocab_name: str) -> None:
        """
        Loads a vocabulary from a file.

        Searches for the vocabulary file in the output directory and loads it into memory.

        Args:
            vocab_name (str): The name of the vocabulary file to load.

        Returns:
            None
        """
        self.__vocab = torch.load(join(VOCABS_DIR, f"{vocab_name}.pth"))

    def get_vocab(self) -> Vocab:
        """
        Returns the vocabulary.

        Returns:
            Vocab: The vocabulary.
        """
        return self.__vocab

    def __normalise(self, tokens: List[str]) -> List[str]:
        """
        Normalises a list of tokens.

        Applies either lemmatisation or stemming to a list of tokens. The normalisation method is set during the
        initialisation of the class, as either "lemmatise" or "stem".

        Args:
            tokens (List[str]): The list of tokens to normalise.
        """
        if self.__normalisation_method == "lemmatise":
            tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens]
        elif self.__normalisation_method == "stem":
            tokens = [PorterStemmer().stem(token) for token in tokens]
        return tokens

    def standardise(self, text: str) -> List[str]:
        """
        Standardises the given text.

        Applies a series of text processing techniques to the given text. The techniques are set during the
        initialisation of the class. The standardisation process is as follows:

            1. Lowercase the text.
            2. Expand contractions.
            3. Remove punctuation.
            4. Tokenise the text.
            5. Remove stop words.
            6. Normalise the tokens.

        Args:
            text (str): The text to standardise.

        Returns:
            List[str]: The standardised tokens.
        """

        if self.__lower_case:
            text = text.lower()

        if self.__expand_contractions:
            text = contractions.fix(text)

        if self.__remove_punctuation:
            text = "".join([char for char in text if char not in string.punctuation])

        tokens: List[str] = word_tokenize(text=text, language="english")

        if self.__remove_stop_words:
            tokens = [token for token in tokens if token not in self.__stop_words]

        if self.__perform_normalisation:
            tokens = self.__normalise(tokens)

        return tokens[:self.__max_tokens]

    def standardise_batch(self, batch: List[str]) -> List[List[str]]:
        """
        Standardises a batch of text.

        Performs the standardisation process on a batch of text by passing each text through the `standardise` method.

        Args:
            batch (List[str]): The batch of text to standardise.

        Returns:
            List[List[str]]: The standardised tokens.
        """
        return [self.standardise(text) for text in batch]

    def vocabularise(self, tokens: List[str], create_vocab: bool = False) -> List[int]:
        """
        Vectorises a list of tokens based on their index in the vocabulary.

        Args:
            tokens (List[str]): The list of tokens to vectorise.
            create_vocab (bool, optional): Whether to create a vocabulary if one does not exist. Defaults to False.

        Returns:
            List[int]: The vectorised tokens.

        Raises:
            ValueError: If no vocabulary exists and `create_vocab` is set to False.
        """

        if self.__vocab is None:
            if create_vocab:
                self.create_vocab(tokens)
            else:
                raise ValueError("No vocabulary has been created. Please create a vocabulary before vocabularising.")

        return [self.__vocab[token] for token in tokens]

    def vocabularise_batch(self, batch: List[List[str]]) -> List[List[int]]:
        """
        Vectorises a batch of tokens based on their index in the vocabulary.

        Performs the vocabularisation process on a batch of tokens by passing each list of tokens through the
        `vocabularise` method.

        Args:
            batch (List[List[str]]): The batch of tokens to vectorise.

        Returns:
            List[List[int]]: The vectorised tokens.

        Raises:
            ValueError: If no vocabulary exists.
        """

        if self.__vocab is None:
            raise ValueError("No vocabulary has been created. Please create a vocabulary before vocabularising.")

        return [self.vocabularise(tokens) for tokens in batch]

    def multi_hot_encode(self, tokens: List[str]) -> ndarray:
        """
        Encodes a list of tokens into a multi-hot vector.

        Args:
            tokens (List[str]): The list of tokens to encode.

        Returns:
            ndarray: The encoded vector.
        """

        if self.__vocab is None:
            raise ValueError("No vocabulary has been created. Please create a vocabulary before encoding.")
        encoded: ndarray = zeros(len(self.__vocab), dtype=int)
        encoded[[self.__vocab[str(token)] for token in tokens]] = 1
        return encoded

    def one_hot_encode(self, tokens: List[str]) -> ndarray:
        raise NotImplementedError

    def encode_batch(self, batch: List[List[str]]) -> ndarray:
        """
        Encodes a batch of tokens.

        Performs the encoding process on a batch of tokens by passing each list of tokens through the `encode` method.
        The encoding method is set during the initialisation of the class, as either "multi-hot" or "one-hot".

        Args:
            batch (List[List[str]]): The batch of tokens to encode.

        Returns:
            ndarray: The encoded vectors.
        """
        if self.__encoding_method == "multi-hot":
            return vstack([self.multi_hot_encode(tokens) for tokens in batch])
        elif self.__encoding_method == "one-hot":
            return vstack([self.one_hot_encode(tokens) for tokens in batch])

    def preprocess_dataset_dict(self,
                                dataset_dict: DatasetDict[str, Dataset],
                                column: str = "text",
                                inplace: bool = True,
                                create_vocab: bool = True,
                                vocab_name: Optional[str] = None) -> DatasetDict[str, Dataset]:
        """
        Preprocesses a DatasetDict of train, validation and test datasets that contain a column named "text".

        This method will standardise the text in the "text" column and then vocabularise the standardised text using
        the vocabulary created by the `create_vocab` method. If a vocabulary has not been created, this method will
        create one using the "train" and "validation" datasets.

        The vocabularised tokens are stored in a new column named "ids", and the original specified column is removed.

        The `DatasetDict` is converted to a torch format before being returned, with only the "ids" and "label" columns
        being preserved.

        Notes:
            - This method assumes that if the `__vocab` attribute is None, then a vocabulary has not been created and
              therefore the "train" and "validation" datasets will be used to create the vocabulary. Realistically,
              the "test" dataset should not be used to create the vocabulary, as this would introduce a bias into the
              model.
            - If a vocabulary has not been created, this method will create one using the "train" and "validation"
              datasets.
            - The vocabulary is saved to a file in the output directory.

        Args:
            dataset_dict (DatasetDict): The DatasetDict to preprocess; each dataset must contain a column named "text".
            vocab_name (Optional[str], optional): The name of the vocabulary file to save. If None, the vocabulary will
                not be saved. Defaults to None.
            column (str, optional): The name of the column to preprocess. Defaults to "text".
            inplace (bool, optional): Whether to perform the preprocessing in place. If True, the specified column will
                be replaced with the preprocessed column. If False, the preprocessed column will be added to the
                DatasetDict. Defaults to True.
            create_vocab (bool, optional): Whether to create a vocabulary if one does not exist. Defaults to True.

        Returns:
            DatasetDict: The preprocessed DatasetDict.
        """

        dataset_dict = dataset_dict.map(
            function=lambda example: {"tokens": self.standardise_batch(example[column])},
            remove_columns=[column] if inplace else None,
            batched=True
        )

        if self.__vocab is None:
            if create_vocab:
                self.create_vocab(chain(dataset_dict["train"]["tokens"], dataset_dict["validation"]["tokens"]))
            else:
                raise ValueError("No vocabulary has been created. Please create a vocabulary before vocabularising.")

        if vocab_name is not None:
            self.save_vocab(vocab_name)

        dataset_dict = dataset_dict.map(
            function=lambda example: {"ids": self.vocabularise_batch(example["tokens"])},
            remove_columns=["tokens"] if inplace else None,
            batched=True
        )

        if self.__encode:
            dataset_dict = dataset_dict.map(
                lambda example: {"ids": self.encode_batch(example["ids"])},
                batched=True
            )

        dataset_dict.set_format(type="torch", columns=["ids", "label"])
        return dataset_dict
