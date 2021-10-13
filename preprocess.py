import logging
import string
import numpy as np
from numpy.core.records import array


class Preprocess:
    """
    A class used to preprocess text data and prepare it as the inputs of skip-gram model and
    also calculate negative sampling distribution

    ...

    Attributes
    ----------
    positional args:
        *files : list
            list of text files
    keyword args:
        **vocab_size : int
            number of frequent words to keep the common words
    txt : str
        line of a text file

    Methods
    -------
    get_word_counts()
        Counts the words frequency throughout the files
    get_vocab_size()
        Gets the minimum of vacab size and unique word counts from the files
    get_word2idx()
        Creates a dictionary from frequent words as keys and indices as values
    sent2idx()
        Replaces words in files with the corresponding index from word2idx dictionary and 
        add each indexed sentance as a list to another list
    get_negative_sampling_distribution()
        Calculates modified unigram distribution of top words
    """
    
    def __init__(self, *files: list, **kwargs: dict):
        """Creates a logging object and get class arguments.

        Positional Args:
            *files(list): list of text files
        Keyword Args:
            **vocab_size(int): number of frequent words to keep the common words
        """

        # Create logging object
        self.logger = logging.getLogger(__name__)
        
        self.files = files
        self.vocab_size = kwargs['vocab_size']
        self.word_counts = {}
        self.sentences = []

    def _remove_punctuation(self, txt: str) -> str:
        """Removes punctuations from text.
        
        The maketrans() method returns a mapping table that can be used with the translate() method 
        to remove specified characters, punctuation here.
        The maketrans() method itself returns a dictionary describing each replacement, in unicode.

        Args:
            txt(str): a text file

        Returns:
            str: a text file after removing punctuations
        """
        
        table = str.maketrans('', '', string.punctuation)
        return txt.translate(table)

    def get_word_counts(self) -> str:
        """Counts the words frequency throughout the files.

        For each line of the file from the files:
        Remove any lines starting with special character,
        Remove punctuations,
        Lower case,
        Tokenize words into a list,
        Add to word_counts dictionary if not existed or
        add one to the frequency if already existed: {word: frequency},
        Create a list containing typles from (key, val) of the dictonary in descending order of word frequencies.
        
        Returns:
            str: list of descending sorted tuples of (word, frequency):
            [('the', 112), ('of', 105), ...]
        """

        for file in self.files:
            for line in open(file):
                
                # Remove the line if it starts with special character because it is a title or metadata
                if line and line[0] not in r'[*-|=\{\}':
                    
                    # Tokenize the line after removeing punctuations and lowering case
                    s = self._remove_punctuation(line).lower().split()

                    if len(s) > 1:
                        for word in s:
                            if word not in self.word_counts:
                                self.word_counts[word] = 0
                            self.word_counts[word] += 1
        self.word_counts = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        return self.word_counts

    def get_vocab_size(self) -> int:
        """Gets the minimum of vacab size and unique word counts from the files.

        Returns:
            int: vocab size
        """
        
        self.vocab_size = min(self.vocab_size, len(self.word_counts))
        return self.vocab_size

    def get_word2idx(self) -> dict:
        """Creates a dictionary from frequent words as keys and indices as values
        plus '<UNK>': 999 for infrequent words.

        Returns:
            dict: dictionary of top frequent words with corresponding index:
            {'the': 0, 'of': 1, ..., '<UNK>': 999}
        """
        
        top_words = [w for w, count in self.word_counts[:self.vocab_size-1]] + ['<UNK>']  # <UNK> for infrequent words
        self.word2idx = {w: i for i, w in enumerate(top_words)}
        return self.word2idx

    def sent2idx(self) -> list:
        """Replaces words in files with the corresponding index from word2idx dictionary and 
        add each indexed sentance as a list to another list.

        Returns:
            list: list of indexed sentances lists:
            [[232, 999, 999, 262], ..., [999, 456, 999]]
        """
        
        for file in self.files:
            for line in open(file):
                s = self._remove_punctuation(line).lower().split()
                
                if len(s) > 1:
                    sentence = [self.word2idx[w] if w in self.word2idx else self.word2idx['<UNK>'] for w in s]
                    self.sentences.append(sentence)
        
        self.logger.info(f'Number of sentences: {len(self.sentences)}')
        return self.sentences

    def get_negative_sampling_distribution(self) -> np.ndarray:
        """Calculates modified unigram distribution of top words.

        This distribution will be used to choose negative samples.
        
        Returns:
            np.ndarray: smoothed unigram distribution for top words
        """
        
        word_freq = np.zeros(self.vocab_size)
        
        for sentence in self.sentences:
            for word in sentence:
                word_freq[word] += 1

        # Smooth it
        p_neg = word_freq ** 0.75
        
        # Normalize it
        self.p_neg = p_neg / p_neg.sum()
        return self.p_neg
