import os
import datetime
import logging
import json
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from preprocess import Preprocess
from skip_gram_tensorflow import SkipGramTensorflow


class Word2Vec:
    """A class used to convert frequent words of text files into corresponding embedding vectors
    and plot t-SNE

    Methods
    -------
    load_config()
        Loads config file
    read_files()
        Returns all text file paths from the file dir
    """
    
    def __init__(self):
        # Create logging object
        self.d = datetime.datetime.now().strftime('%Y-%m-%d--%H:%M')
        self.log_dir = os.path.join('logs', f'{self.d}.log')
        self._get_logger()
        self.logger = logging.getLogger(__name__)
        self.logger.info('Created a log file')
        
        # Config file path
        self.config_path = 'configs/config.json'

        # Data file directory
        self.file_dir = '*.txt'

    def _get_logger(self):
        """Creates a logging file.
        """
        
        if not os.path.exists('logs'):
            os.makedirs('logs')

        # Create and configure logger with the threshold to INFO
        logging.basicConfig(filename=self.log_dir,
                            level=logging.INFO,
                            format='%(asctime)s — %(name)s - %(levelname)s: %(message)s',
                            filemode='w')

    def load_config(self) -> json:
        """Loads config file.

        Returns:
            json: JSON file of config values for classes
        """
        
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            self.logger.error('Config file not exists')
            exit()

    def read_files(self):
        """Returns all text file paths from the file path.
        
        """
        
        files = glob(self.file_dir)
        self.logger.info(f'Number of files: {len(files)}')
        return files
    
    def tsne(self, word2idx: dict, embedding_dict: dict, vocab_size: int) -> None:
        """Creates 2-dim space T-distributed Stochastic Neighbor Embedding (t-SNE) plot from word embeddings.

        Args:
            word2idx(dict):
                dictionary of (top frequent words, corresponding index)
            embedding_dict(dict):
                dictionary of (word, embedding list)
            vocab_size(int):
                number of frequent words to keep the common words
        """
        
        # Create a dictionary of (index, word)
        idx2word = {idx:word for idx, word in enumerate(word2idx)}
        
        tsne = TSNE()
        Z = tsne.fit_transform(np.array(list(embedding_dict.values())))
        plt.scatter(Z[:,0], Z[:,1])
        plt.title('t-SNE visualization')
        
        for i in range(vocab_size):
            try:
                plt.annotate(s=idx2word[i], xy=(Z[i,0], Z[i,1]))
            except:
                print('bad string:', idx2word[i])
        plt.show()


def main():
    W2V = Word2Vec()
    config = W2V.load_config()
    
    # Get data files
    files = W2V.read_files()
    
    # Preprocess files
    PRE = Preprocess(*files, **config['preprocess'])
    PRE.get_word_counts()
    vocab_size = PRE.get_vocab_size()
    word2idx = PRE.get_word2idx()
    sentences = PRE.sent2idx()
    p_neg = PRE.get_negative_sampling_distribution()
    
    # Train Skip-Gram model using Tensorflow library
    SGT = SkipGramTensorflow(sentences, p_neg, vocab_size, word2idx, **config['skip_gram'])
    embedding_dict =  SGT.train_model()

    # Plot t-SNE
    W2V.tsne(word2idx, embedding_dict, vocab_size)

if __name__ == "__main__":
    main()
