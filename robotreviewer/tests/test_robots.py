import gc
import json
import numpy as np
import os
import unittest

import robotreviewer
from robotreviewer.data_structures import MultiDict
from robotreviewer.robots.pico_robot import PICORobot
from robotreviewer.robots.pico_viz_robot import PICOVizRobot
from robotreviewer.robots.pubmed_robot import PubmedRobot
from robotreviewer.robots.rct_robot import RCTRobot


class TestPICORobot(unittest.TestCase):
        
    pr = PICORobot()
    ex_file = os.path.dirname(__file__) + "/ex/pico.json"
        
    def test_get_positional_features(self):
        ''' test for PICORobot._get_positional_features(sentences) '''
        with open(self.ex_file) as data:
            data = json.load(data)
        before = data["before"]
        after = data["after"]
        test = PICORobot._get_positional_features(before)
        self.assertEqual(test, after)

    def test_token_contains_number(self):
        ''' test for PICO_vectorizer.token_contains_number(token) '''
        i1, o1 = "the", False
        self.assertEqual(self.pr.vec.token_contains_number(i1), o1)
        i2, o2 = "2013.09.032", True
        self.assertEqual(self.pr.vec.token_contains_number(i2), o2)
        i3, o3 = "http://dx.doi.org/10.1016/j.vaccine.", True
        self.assertEqual(self.pr.vec.token_contains_number(i3), o3)

class TestPICOVizRobot(unittest.TestCase):
    
    pv = PICOVizRobot()
    ex_path = os.path.dirname(__file__) + "/ex/"
        
    def test_postprocess_embedding(self):
        ''' test for PICOVizRobot.postprocess_embedding(H) '''
        before = np.load(self.ex_path + "before.npy")
        after = np.load(self.ex_path + "after.npy")
        test = self.pv.postprocess_embedding(before)
        self.assertTrue(np.array_equal(after, test))
        gc.collect()
        
    def test_tokenize(self):
        ''' test for PICOVizRobot.tokenize(text) '''
        with open(self.ex_path + "pico_viz.json") as datafile:
            data = json.load(datafile)
        test = data["token_start"]
        tok = self.pv.tokenize(test)
        end = data["token_end"]
        self.assertEqual(tok, end)
        gc.collect()
        
    def test_annotate(self):
        ''' test for PICOVizRobot.annotate(data) '''
        with open(self.ex_path + "pico_viz.json") as datafile:
            data = json.load(datafile)
        md = MultiDict()
        md.data["gold"]["abstract"] = data["abstract"]
        md = self.pv.annotate(md)
        
        p_vector = data["p_vector"]
        self.assertEqual(md.data["ml"]["p_vector"], p_vector)
        p_words = data["p_words"]
        self.assertEqual(md.data["ml"]["p_words"], p_words)
        i_vector = data["i_vector"]
        self.assertEqual(md.data["ml"]["i_vector"], i_vector)
        i_words = data["i_words"]
        self.assertEqual(md.data["ml"]["i_words"], i_words)
        o_vector = data["o_vector"]
        self.assertEqual(md.data["ml"]["o_vector"], o_vector)
        o_words = data["o_words"]
        self.assertEqual(md.data["ml"]["o_words"], o_words)
        gc.collect()

class TestPubmedRobot(unittest.TestCase):

    pr = PubmedRobot()
    ex_file = os.path.dirname(__file__) + "/ex/pubmedtest.json"

    def test_annotate(self):
        ''' test for PubmedRobot.annotate(data) '''
        md = MultiDict()
        with open(self.ex_file) as testdata:
            data = json.load(testdata)
        test = data["annotate"]
        md.data["gold"]["title"] = data["title"]
        out = self.pr.annotate(md)
        self.assertEqual(out.data["pubmed"], test)

    def test_query_pubmed(self):
        ''' test for PubmedRobot.query_pubmed(pmid) '''
        with open(self.ex_file) as testdata:
            data = json.load(testdata)
        pmid = data["pmid"]
        q = self.pr.query_pubmed(pmid)
        with open(os.path.dirname(__file__) + "/ex/query.json") as f:
            query = json.load(f)
        self.assertEqual(q, query)

    def test_short_citation(self):
        ''' test for PubmedRobot.short_citation(data) '''
        data = {
            "authors": [
                {"lastname": "Bellman", "initials": "R"},
                {"lastname": "Ford", "initials": "L"}
            ],
            "year": 1958
        }
        short_cite = self.pr.short_citation(data)
        self.assertEqual(short_cite, "Bellman R, 1958")

class TestRCTRobot(unittest.TestCase):
    
    rct = RCTRobot()
    ex_path = os.path.dirname(__file__) + "/ex/"
    
    def test_annotate(self):
        ''' test for RCTRobot.annotate(data) '''
        with open(self.ex_path + "rct.json") as data:
            data = json.load(data)
        md = MultiDict()
        md.data["gold"]["title"] = data["title"]
        md.data["gold"]["abstract"] = data["abstract"]
        md.data["pubmed"] = True
        md = self.rct.annotate(md)
        test = {'is_rct': True, 'model_class': 'svm_cnn_ptyp', 'decision_score': 7.7760185186526991}
        self.assertEqual(md.ml["rct"], test)
        gc.collect()
        
    def test_kv_transform(self):
        ''' test for KerasVectorizer.transform(raw_documents) '''
        with open(self.ex_path + "rct.json") as data:
            data = json.load(data)
        kv = self.rct.cnn_vectorizer
        raw_documents = data["raw_documents"]
        test = np.load(self.ex_path + "kv_transform.npy")
        self.assertTrue(np.array_equal(kv.transform(raw_documents), test))
        gc.collect()
