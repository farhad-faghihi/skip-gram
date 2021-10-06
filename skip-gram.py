import numpy as np
import pandas as pd
import tensorflow as tf
from glob import glob
import string
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class SkipGram:
    """
    A class used to convert each word from a corpus to the corresponding numerical embedding vector using word2vec (skip-gram) model
    """
    def __init__(self):
        self.file_path = '*.txt'
        self.vocab_size = 1000
        self.context_size = 5
        self.num_negatives = 5  # Number of negative samples to draw per target
        self.learning_rate = 0.01
        self.epochs = 10
        self.embedding_size = 100
        self.word_counts = {}
        self.sentences = []
        self.losses = []
        self.embedding_dict = {}

    def _remove_punctuation(self, s):
        table = str.maketrans('','', string.punctuation)
        return s.translate(table)

    def _get_word_counts(self):
        files = glob(self.file_path)

        for file in files:
            for line in open(file):
                s = self._remove_punctuation(line).lower().split()

                if len(s) > 1:
                    for word in s:
                        if word not in self.word_counts:
                            self.word_counts[word] = 0
                        self.word_counts[word] += 1
        self.word_counts = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        return self.word_counts

    def _word2idx(self):
        """
        Return dictionary of top frequent words with corresponding index
        """
        top_words = [w for w, count in self.word_counts[:self.vocab_size-1]] + ['<UNK>']  # UNK for infrequent words
        self.word2idx = {w: i for i, w in enumerate(top_words)}
        return self.word2idx

    def _sent2idx(self):
        files = glob(self.file_path)

        for file in files:
            for line in open(file):
                s = self._remove_punctuation(line).lower().split()
                if len(s) > 1:
                    sentence = [self.word2idx[w] if w in self.word2idx else self.word2idx['<UNK>'] for w in s]
                    self.sentences.append(sentence)
        return self.sentences
        
    def _get_negative_sampling_distribution(self):
        """
        Return smoothed unigram distribution for top words
        """
        word_freq = np.zeros(self.vocab_size)
        for word in self.sentences:
            # for word in sentence:
            word_freq[word] += 1

        # Smooth it
        self.p_neg = word_freq ** 0.75
        
        # Normalize it
        self.p_neg = self.p_neg / self.p_neg.sum()
        # print('\n p_neg:', self.p_neg)
        return self.p_neg

    def _init_weights(self):
        """
        Initialize weights to random normal distributed numbers
        """
        self.W1 = tf.Variable(tf.random.normal([self.vocab_size, self.embedding_size]))  # input to embedding
        # print('\n W1: \n', self.W1)
        self.W2 = tf.Variable(tf.random.normal([self.embedding_size, self.vocab_size]))  # embedding to output
        # print('\n W2: \n', self.W2)
        return self.W1, self.W2 

    def _optimizer(self):
        """
        Create Adam optimzer
        """
        return tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def _get_context(self, position, sentence):
        start = max(0, position - self.context_size)
        end = min(len(sentence), position + self.context_size)

        contexts = []
        for context_position, word_idx in enumerate(sentence[start:end], start=start):
            if context_position != position:
                # Don't include the input word itself as a target
                contexts.append(word_idx)
        return contexts

    def _feedforward(self, target, contexts, neg_word):
        """
        Predict correct and incorect contexts in feedfowrd direcion (N: context size, D: embedding dimension)
        """
        # Correct word output
        emb_input = tf.nn.embedding_lookup(self.W1, target)  # 1 x D
        # print('\n embedding input:', emb_input)
        emb_output = tf.nn.embedding_lookup(tf.transpose(self.W2), contexts)  # N x D
        # print('\n embedding output:', emb_output)
        correct_output = tf.math.reduce_sum(emb_input * emb_output, axis=1)  # N
        # print('\n correct output:', self.correct_output)
        
        # Incorrect word output
        emb_input = tf.nn.embedding_lookup(self.W1, neg_word)
        incorrect_output = tf.math.reduce_sum(emb_input * emb_output, axis=1)
        return correct_output, incorrect_output

    def _loss(self, correct_output, incorrect_output):
        """
        Calculate loss function
        """
        pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones(tf.shape(correct_output)),
                                                           logits=correct_output)

        neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros(tf.shape(incorrect_output)),
                                                           logits=incorrect_output)
        # Total loss
        loss = tf.math.reduce_mean(pos_loss) + tf.math.reduce_mean(neg_loss)
        # print('\n loss:', loss)
        return loss

    def _train_one_step(self, target, contexts):
        with tf.GradientTape() as g:
            neg_word = np.random.choice(self.vocab_size, p=self.p_neg)  # Instead of changing context words, target word changes for negative sampling
            correct_output, incorrect_output = self._feedforward(target, contexts, neg_word)
            loss = self._loss(correct_output, incorrect_output)
        gradients = g.gradient(loss, [self.W1, self.W2])
        self.optimizer.apply_gradients(zip(gradients, [self.W1, self.W2]))
        self.loss.update_state(loss)

    def _embedding_dict(self):
        """
        Create dictionary, words as keys and embedding vectors as values
        """
        for i in range(self.vocab_size):
            self.embedding_dict[list(self.word2idx.keys())[i]] = self.W1.numpy().tolist()[i]
        # print('embedding_dict: \n', self.embedding_dict)
        return self.embedding_dict

    def train_model(self):
        self.word_counts = self._get_word_counts()
        self.word2idx = self._word2idx()
        self.sentences = self._sent2idx()
        self.p_neg = self._get_negative_sampling_distribution()
        self.W1, self.W2 = self._init_weights()
        self.optimizer = self._optimizer()
        
        # Compute the mean of the given losses
        self.loss = tf.keras.metrics.Mean()
        
        for epoch in range(self.epochs):
            for sentence in self.sentences:
                for position, target in enumerate(sentence):
                    # Get the positive context words/negative samples
                    contexts = self._get_context(position, sentence)
                    self._train_one_step(target, contexts)

            # Save the loss
            self.losses.append(self.loss)

            print('epoch:', epoch, 'loss:', self.loss.result().numpy())
        
        self.embedding_dict = self._embedding_dict()
        return self.embedding_dict

    def tsne(self):
        """
        Create two_dimensional T-distributed Stochastic Neighbor Embedding (t-SNE) plot from word embeddings
        """
        idx2word = {idx:word for idx, word in enumerate(self.word2idx)}
        tsne = TSNE()
        Z = tsne.fit_transform(self.W1.numpy())
        plt.scatter(Z[:,0], Z[:,1])
        plt.title('t-SNE visualization')
        for i in range(self.vocab_size):
            try:
                plt.annotate(s=idx2word[i], xy=(Z[i,0], Z[i,1]))
            except:
                print('bad string:', idx2word[i])
        plt.show()


def main():
    skipgram = SkipGram()
    skipgram.train_model()
    skipgram.tsne()

if __name__ == '__main__':
    main()
