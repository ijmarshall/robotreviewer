import json
import os
import unittest

import numpy as np
from scipy.sparse import csr_matrix

import robotreviewer
from robotreviewer.ml.classifier import MiniClassifier
from robotreviewer.ml.vectorizer import ModularVectorizer
from robotreviewer.ml.vectorizer import InteractionHashingVectorizer
from robotreviewer.ml.vectorizer import Vectorizer

from robotreviewer.tests import ex_path

class Utilities(object):

    def save_sparse_csr(self, filename, array):
        np.savez(ex_path + filename, data=array.data,
                 indices=array.indices, indptr=array.indptr, shape=array.shape)

    def load_sparse_csr(self, filename):
        loader = np.load(ex_path + filename)
        return csr_matrix((loader['data'], loader['indices'],
                          loader['indptr']), shape=loader['shape'])

class TestMiniClassifier(unittest.TestCase):

    doc_clf = MiniClassifier(robotreviewer.get_data('bias/bias_doc_level.npz'))
    util = Utilities()

    def test_init(self):
        ''' test for MiniClassifier.__init__() '''
        self.assertTrue(isinstance(self.doc_clf.coef, np.ndarray))
        self.assertTrue(isinstance(self.doc_clf.intercept, float))

    def test_decision_function(self):
        ''' test for MiniClassifier.decision_function(X) '''
        X = self.util.load_sparse_csr("X_data.npz")
        dec = self.doc_clf.decision_function(X)  # [ 1.50563252]
        decTest = np.float64([1.50563252])
        ''' can't do:
            print(np.array_equal(dec, y))
            print(np.array_equiv(dec, y))
            since as decimals these will not pass
        '''
        self.assertTrue(np.allclose(dec, decTest))

    def test_predict(self):
        ''' test for MiniClassifier.predict(X) '''
        X = self.util.load_sparse_csr("X_data.npz")
        pred = self.doc_clf.predict(X)  # [1]
        self.assertEqual(pred, np.int(1))

    def test_predict_proba(self):
        ''' tests for MiniClassifier.predict_proba(X) '''
        with open(ex_path + "rationale_robot_data.json", "r", encoding="utf-8") as data:
            data = json.load(data)
        bpl = data["bias_prob_linear"]
        X = self.util.load_sparse_csr("X_data.npz")
        bpl_test = self.doc_clf.predict_proba(X)[0]
        self.assertTrue(abs(bpl - bpl_test) < 0.01)

class TestModularVectorizer(unittest.TestCase):
    
    util = Utilities()
    m = ModularVectorizer(norm=None, non_negative=True, binary=True,
                          ngram_range=(1, 2), n_features=2**26)
    
    def test_init(self):
        ''' test for ModularVectorizer.__init__() '''
        m = ModularVectorizer(norm=None, non_negative=True, binary=True,
                              ngram_range=(1, 2), n_features=2**26)
        self.assertTrue(m.vec is not None)
        self.assertEqual(type(m.vec), InteractionHashingVectorizer)
        
    def test_combine_matrices(self):
        ''' test for ModularVectorizer.combine_matrices(X_part) '''
        self.m.builder_clear()
        X_part = self.util.load_sparse_csr("vec_builder.npz")
        self.m._combine_matrices(X_part)
        X_part.data.fill(1)
        self.assertEqual((X_part != self.m.X).nnz, 0)
        X_part2 = self.util.load_sparse_csr("vec_builder.npz")
        self.m._combine_matrices(X_part2)
        save = X_part + X_part2
        self.assertEqual((save != self.m.X).nnz, 0)
        
    def test_builder_clear(self):
        ''' test for ModularVectorizer.builder_clear() '''
        self.m.builder_clear()
        self.assertTrue(self.m.X is None)
        self.m.X = ["anything"]
        self.m.builder_clear()
        self.assertTrue(self.m.X is None)
        
    def test_builder_add_docs(self):
        ''' test for ModularVectorizer.builder_add_docs() '''
        self.m.builder_clear()
        with open(ex_path + "vector_si.json") as data:
            data = json.load(data)
        X_si = [(data["X_si0"], data["X_si1"])]
        self.assertEqual(self.m.X, None)
        self.m.builder_add_docs(X_si)
        self.assertTrue(self.m.X is not None)
        
    def test_builder_transform(self):
        ''' test for  ModularVectorizer.builder_transform '''
        self.m.builder_clear()
        self.assertEqual(self.m.builder_transform(), None)
        self.m.X = ["anything"]
        self.assertTrue(self.m.builder_transform(), ["anything"])

class TestInteractionHashingVectorizer(unittest.TestCase):
    
    util = Utilities()
    
    def test_transform(self):
        ''' test for InteractionHashingVectorizer.transform(X_si) '''
        ih = InteractionHashingVectorizer(norm=None, non_negative=True,
                                          binary=True, ngram_range=(1, 2),
                                          n_features=2**26)
        with open(ex_path + "vector_si.json") as data:
            data = json.load(data)
        X_si = [(data["X_si0"], data["X_si1"])]
        X_part_Test = ih.transform(X_si)
        X_part = self.util.load_sparse_csr("vec_builder.npz")
        self.assertEqual((X_part != X_part_Test).nnz, 0)

class TestVectorizer(unittest.TestCase):
    
    def test_init(self):
        ''' test for Vectorizer.__init__() '''
        v = Vectorizer()
        self.assertEqual(v.embeddings, None)
        self.assertEqual(v.word_dim, 300)
        
    def test_fit(self):
        ''' test for Vectorizer.fit(texts) '''
        with open(ex_path + "vector_tests.json", "r", encoding="utf-8") as data:
            data = json.load(data)
        v = Vectorizer()
        v.fit(data["text"])
        self.assertEqual(v.texts, data["text"])
