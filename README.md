# Word2Vec - Skip-Gram Model

Word2Vec is a language model that are used to produce word embeddings that was developed by a team of researchers led by Tomas Mikolov at Google in 2013. This model turns turn one-hot encoded vectors of words to dense vectors. While Word2vec is a two-layer neural network, the embedding vectors created by this model can be used as the numerical input of deep neural networks. There are two model architectures of Word2Vec: Continuous Bag-Of-Words (CBOW) and Skip-Gram. CBOW: several times faster to train than the Skip-Gram, slightly better accuracy for the frequent words. Skip-Gram works well with small amount of the training data, represents well even rare words or phrases
SKip-Gram model tries to predict the context words (surrounding words) for given a target word (the center word). Skip-gram model is trained with negative sampling as it is faster and works better for frequent words. Negative samples are selected randomly based on smoothed unigram distribution. 

**Negative sampling:** Unigram distribution is the porbability of a single word occuring. The problem with this distribution is that infrequent words are too infrequent so they are very unlikely to be sampled. We can smooth that distribution by raising all of our accounts by 0.75.

**t-SNE:** T-distributed Stochastic Neighbor Embedding (t-SNE) is a machine learning algorithm for visualization technique well-suited for embedding high-dimensional data into two or three dimensions. It reduces the dimensionality of data while it preserves the structure of data. 

## Requirements

python 3.7
conda 4.8

## Installing the environment

To create the virtual environment run the following command:
conda env create -f environment.yml

Activate the new environment: 
conda activate myenv

Verify that the new environment was installed correctly:
conda env list

## Clone

Clone this repo to your local machine using https://github.com/farhad-faghihi/skip-gram

## Run

In a terminal or command window, navigate to the top-level project directory skip-gram/ and run the following commands:
python skip-gram.py

## Data

You can save your text files with as .txt extension inside the top-level project directory.
