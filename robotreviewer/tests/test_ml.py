
import json
import os
import unittest

import numpy as np
from scipy.sparse import csr_matrix

import robotreviewer
from robotreviewer.ml.classifier import MiniClassifier

class Utilities(object):

    ex_path = os.path.dirname(__file__) + "/ex/"

    def save_sparse_csr(self, filename, array):
        np.savez(self.ex_path + filename, data=array.data,
                 indices=array.indices, indptr=array.indptr, shape=array.shape)

    def load_sparse_csr(self, filename):
        loader = np.load(self.ex_path + filename)
        return csr_matrix((loader['data'], loader['indices'],
                          loader['indptr']), shape=loader['shape'])


class TestMiniClassifier(unittest.TestCase):

    doc_clf = MiniClassifier(robotreviewer.get_data('bias/bias_doc_level.npz'))
    util = Utilities()

    def test_init(self):
        ''' test for MiniClassifier.__init__() '''
        self.assertEqual(isinstance(self.doc_clf.coef, np.ndarray), True)
        self.assertEqual(isinstance(self.doc_clf.intercept, float), True)

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
        self.assertEqual(np.allclose(dec, decTest), True)

    def test_predict(self):
        ''' test for MiniClassifier.predict(X) '''
        X = self.util.load_sparse_csr("X_data.npz")
        pred = self.doc_clf.predict(X)  # [1]
        self.assertEqual(pred, np.int(1))

    def test_predict_proba(self):
        ''' tests for MiniClassifier.predict_proba(X) '''
        with open(self.util.ex_path + "rationale_robot_data.json") as data:
            data = json.load(data)
        bpl = data["bias_prob_linear"]
        X = self.util.load_sparse_csr("X_data.npz")
        bpl_test = self.doc_clf.predict_proba(X)[0]
        self.assertEqual(abs(bpl - bpl_test) < 0.01, True)
