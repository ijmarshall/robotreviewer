"""
low memory modular vectorizer for multitask learning
"""

# Authors:  Iain Marshall <mail@ijmarshall.com>
#           Joel Kuiper <me@joelkuiper.com>
#           Byron Wallce <byron.wallace@utexas.edu>

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

import numpy as np
import scipy


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



