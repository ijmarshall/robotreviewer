"""
low memory modular vectorizer for multitask learning
"""

# Authors:  Iain Marshall <mail@ijmarshall.com>
#           Joel Kuiper <me@joelkuiper.com>
#           Byron Wallace <byron@ccs.neu.edu>

import pickle

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer


class ModularVectorizer(object):

    def __init__(self, *args, **kwargs):
        self.vec = InteractionHashingVectorizer(*args, **kwargs)

    def builder_clear(self):
        self.X = None

    def _combine_matrices(self, X_part, weighting=1):
        X_part.data.fill(weighting)

        if self.X is None:
            self.X = X_part
        else:
            self.X = self.X + X_part
            # assuming we have no collisions, the interaction terms shouldn't be identical
            # if there are collisions, this is ok since they should form a tiny proportion
            # of the data (they will have values > weighting)

    def builder_add_docs(self, X_si, weighting=1, low=None):
        X_part = self.vec.transform(X_si, low=low)
        self._combine_matrices(X_part, weighting=weighting)

    def builder_transform(self):
        return self.X


class InteractionHashingVectorizer(HashingVectorizer):
    """
    Same as HashingVectorizer,
    but with an option to add interaction prefixes to the
    tokenized words, and option to take a binary mask vector
    indicating which documents to add interactions for
    """

    def __init__(self, *args, **kwargs):

        # this subclass requires certain parameters - check these

        assert kwargs.get("analyzer", "word") == "word" # only word tokenization (default)
        assert kwargs.get("norm") is None # don't normalise words (i.e. counts only)
        assert kwargs.get("binary") == True 
        assert kwargs.get("non_negative") == True

        super(InteractionHashingVectorizer, self).__init__(*args, **kwargs)

    def build_analyzer(self):
        """Return a callable that handles preprocessing and tokenization"""
        preprocess = self.build_preprocessor()

        # only does word level analysis
        stop_words = self.get_stop_words()
        tokenize = self.build_tokenizer()

        return lambda doc_i: self._word_ngrams(
            tokenize(preprocess(self.decode(doc_i[0]))), stop_words, doc_i[1])

    def _word_ngrams(self, tokens, stop_words=None, i_term=None):
        """
        calls super of _word_ngrams, then adds interaction prefix onto each token
        """
        tokens = super(InteractionHashingVectorizer, self)._word_ngrams(tokens, stop_words)

        if i_term:
            return [i_term + token for token in tokens]
        else:
            return tokens

    def _deal_with_input(self, doc):
        """
        If passed a doc alone, returns a blank interaction
        string. If passed an (doc, i_term) tuple returns 
        (doc, i_term), except if i_term="" then returns
        "", ""
        """
        if isinstance(doc, tuple):
            if doc[1]:
                return doc
            else:
                return ("", "")
        else:
            return (doc, "")

    def transform(self, X_si, high=None, low=None, limit=None):
        """
        Same as HashingVectorizer transform, except allows for 
        interaction list, which is an iterable the same length as X
        filled with True/False. This method adds an empty row to
        docs labelled as False.
        """
        analyzer = self.build_analyzer()

        X = self._get_hasher().transform(
            analyzer(self._deal_with_input(doc)) for doc in X_si)
        
        X.data.fill(1)

        if self.norm is not None:
            X = normalize(X, norm=self.norm, copy=False)

        if low:
            X = self._limit_features(X, low=low)
        return X


class Vectorizer:
    """Tiny class for fitting a vocabulary and vectorizing texts.
    Assumes texts have already been tokenized and that tokens are separated by
    whitespace.
    This class maintains state so that it can be pickled and used to train keras
    models.
    """
    def __init__(self):
        self.embeddings = None
        self.word_dim = 300

    def __len__(self):
        """Return the length of X"""
        return len(self.X)

    def __getitem__(self, given):
        """Return a slice of X"""
        return self.X[given]

    def fit(self, texts, maxlen=None, maxlen_ratio=.95):
        """Fit the texts with a keras tokenizer
        
        Parameters
        ----------
        texts : list of strings to fit and vectorize
        maxlen : maximum length for texts
        maxlen_ratio : compute maxlen M dynamically as follows: M is the minimum
        number such that `maxlen_ratio` percent of texts have length greater
        than or equal to M.
        
        """
        # fit vocabulary
        self.tok = Tokenizer(filters='')
        self.tok.fit_on_texts(texts)

        # set up dicts
        self.word2idx = self.tok.word_index
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        if not maxlen:
            # compute `maxlen` dynamically
            lengths = pd.Series(len(text.split()) for text in texts)
            for length in range(min(lengths), max(lengths)):
                nb_lengths = len(lengths[lengths <= length])
                if nb_lengths / float(len(texts)) >= maxlen_ratio:
                    self.maxlen = length
                    break
        else:
            self.maxlen = maxlen

        self.texts = texts
        self.vocab_size = len(self.word2idx)

    def texts_to_sequences(self, texts, do_pad=True):
        """Vectorize texts as sequences of indices
        
        Parameters
        ----------
        texts : list of strings to vectorize into sequences of indices
        do_pad : pad the sequences to `self.maxlen` if true
        """
        self.X = self.tok.texts_to_sequences(texts)

        if do_pad:
            self.X = sequence.pad_sequences(self.X, maxlen=self.maxlen)
            self.word2idx['[0]'], self.idx2word[0] = 0, '[0]' # add padding token
            self.vocab_size += 1

        return self.X

    def texts_to_BoW(self, texts):
        """Vectorize texts as BoW
        
        Parameters
        ----------
        texts : list of strings to vectorize into BoW
        
        """
        self.X = self.tok.texts_to_matrix(texts)
        self.X = self.X[:, 1:] # ignore the padding token prepended by keras
        self.X = csr_matrix(self.X) # space-saving

        return self.X

    def extract_embeddings(self, model):
        """Pull out pretrained word vectors for every word in vocabulary
        
        Parameters
        ----------
        model : gensim word2vec model
        If word is not in `model`, then randomly initialize it.
        
        """
        self.word_dim, self.vocab_size = model.vector_size, len(self.word2idx)
        self.embeddings = np.zeros([self.vocab_size, self.word_dim])

        for i, word in sorted(self.idx2word.items()):
            self.embeddings[i] = model[word] if word in model else np.random.randn(self.word_dim)

        return self.embeddings

    def test(self, doc_idx):
        """Recover text from vectorized representation of the `doc_idx`th text
        Parameters
        ----------
        doc_idx : document index to recover text from
        This function is just for sanity checking that sequence vectorization
        works.
        """
        print(self.X[doc_idx])
        print()
        print(' '.join(self.idx2word[idx] for idx in self.X[doc_idx] if idx))
