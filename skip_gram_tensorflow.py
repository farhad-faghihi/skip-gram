import logging
from typing import Tuple
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class SkipGramTensorflow:
    """A class used to train skip-gram model to calculate the numerical embedding vector of 
    frequent words in a corpus

    ...

    Attributes
    ----------
    sentences : list
        list of indexed sentances lists
    p_eng : array
        smoothed unigram distribution for top words
    vocab_size : int
        number of frequent words to keep the common words
    keyword args:
        **context_size : int
            number of surrounding words around the target word
        **num_negatives : int
            number of negative samples to draw per target
        **learning_rate : int
            learning rate
        **epochs : int
            number of epochs
        **embedding_size : int
            number of hidden neurons as embedding vocab size

    Methods
    -------
    train_model()
        Trains neural network over epochs and returns embedding dictionary using hidden weight matrix
    """
    
    def __init__(self, sentences, p_neg, vocab_size, word2idx, **kwargs):
        # Create logging object
        self.logger = logging.getLogger(__name__)

        self.sentences = sentences
        self.p_neg = p_neg
        self.vocab_size = vocab_size
        self.word2idx = word2idx
        self.losses = []
        self.embedding_dict = {}

        # Hyperparameters
        self.context_size = kwargs['context_size']
        self.num_negatives = kwargs['num_negatives']
        self.learning_rate = kwargs['learning_rate']
        self.num_epochs = kwargs['num_epochs']
        self.embedding_size = min(kwargs['max_embedding_size'], (vocab_size+1)//2)  # Rule of thumb in calculating the embedding size

    def _init_weights(self):
        """Initializes weights to random normal distributed numbers.
        """
        
        # Input to embedding layer
        self.W1 = tf.Variable(tf.random.normal([self.vocab_size, self.embedding_size]))
        
        # Embedding to output layer
        self.W2 = tf.Variable(tf.random.normal([self.embedding_size, self.vocab_size]))
        return self.W1, self.W2 

    def _optimizer(self):
        """Creates Adam optimzer.
        """
        
        return tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def _get_context(self, position: int, sentence) -> list:
        """Gets surrounding words around the target word that will be used as positive samples of model outputs.
        
        Args:
            position(int):
                position of the target word in the sentence
            sentence(list):
                list of indexed words of a sentence

        Returns:
            list: list of context words around the target word
        """
        
        contexts = []

        start = max(0, position - self.context_size)
        end = min(len(sentence), position + self.context_size + 1)

        for context_position, word_idx in enumerate(sentence[start:end]):

            # Doesn't include the target (input) word itself as a context (output) word
            if context_position != position:
                contexts.append(word_idx)
        return contexts

    def _feedforward(self, target: int, contexts: list, neg_word: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Predicts correct and incorrect contexts in feedfowrd direcion (N: context size, D: embedding dimension).
        
        Args:
            target(int):
                target word
            contexts(list):
                list of context words
            neg_word(int):
                negative word

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: predicted model outputs for correct and incorrect contexts
        """
        
        # Correct word output
        emb_input = tf.nn.embedding_lookup(self.W1, target)  # 1 x D
        emb_output = tf.nn.embedding_lookup(tf.transpose(self.W2), contexts)  # N x D
        correct_output = tf.math.reduce_sum(emb_input * emb_output, axis=1)  # N
        
        # Incorrect word output
        emb_input = tf.nn.embedding_lookup(self.W1, neg_word)  # 1 x D
        incorrect_output = tf.math.reduce_sum(emb_input * emb_output, axis=1)  # N
        return correct_output, incorrect_output

    def _loss(self, correct_output: tf.Tensor, incorrect_output: tf.Tensor) -> tf.Tensor:
        """Calculates loss value from predicted ouputs using binary cross-entropy function.

        Args:
            correct_output(tf.Tensor):
                predicted model outputs for correct contexts
            incorrect_output(tf.Tensor):
                predicted model outputs for incorrect contexts

        Returns:
            tf.Tensor: loss value
        """
        
        pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones(tf.shape(correct_output)),
                                                           logits=correct_output)

        neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros(tf.shape(incorrect_output)),
                                                           logits=incorrect_output)
        # Total loss
        loss = tf.math.reduce_mean(pos_loss) + tf.math.reduce_mean(neg_loss)
        return loss

    def _train_one_step(self, target: int, contexts: list) -> None:
        """Updates neural network weights.
        
        Args:
            target(int):
                target word
            contexts(list):
                list of context words
        """
        with tf.GradientTape() as g:
            
            # Get a random integer between the vocab_size with the probabilities from p_neg
            # Instead of changing context words, target word changes for negative sampling
            neg_word = np.random.choice(self.vocab_size, p=self.p_neg)
            correct_output, incorrect_output = self._feedforward(target, contexts, neg_word)
            loss = self._loss(correct_output, incorrect_output)
        gradients = g.gradient(loss, [self.W1, self.W2])
        self.optimizer.apply_gradients(zip(gradients, [self.W1, self.W2]))
        self.loss.update_state(loss)

    def _embedding_dict(self) -> dict:
        """Creates dictionary, words as keys and embedding vectors as values.

        Returns:
            dict: dictionary of (word, embedding list)
        """
        
        for i in range(self.vocab_size):
            self.embedding_dict[list(self.word2idx.keys())[i]] = self.W1.numpy().tolist()[i]
        return self.embedding_dict

    def train_model(self):
        """Trains neural network over epochs and returns embedding dictionary using hidden weight matrix.
        
        Returns:
            dict: dictionary of (word, embedding list)
        """
        
        self.logger.info('Training started')

        # Initialize weights
        self.W1, self.W2 = self._init_weights()

        # Create optimizer
        self.optimizer = self._optimizer()
        
        # Compute the mean of the given losses
        self.loss = tf.keras.metrics.Mean()
        
        for epoch in range(self.num_epochs):
            for sentence in self.sentences:
                for position, target in enumerate(sentence):
                    
                    # Get the positive context words/negative samples
                    contexts = self._get_context(position, sentence)
                    self._train_one_step(target, contexts)

            # Save the loss
            self.losses.append(self.loss)

            self.logger.info(f'epoch: {epoch}, loss:, {self.loss.result().numpy()}')
        
        self.logger.info('Training finished')

        self.embedding_dict = self._embedding_dict()
        return self.embedding_dict
