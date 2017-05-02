import json
import unittest
import os
import numpy as np

import robotreviewer
from robotreviewer.data_structures import MultiDict
from robotreviewer.robots.rationale_robot import BiasRobot
from robotreviewer.robots.pico_viz_robot import PICOVizRobot
from robotreviewer.robots.pubmed_robot import PubmedRobot

class TestBiasRobot(unittest.TestCase):

    br = BiasRobot()

    def test_simple_borda_count(self):
        ''' tests for BiasRobot.simple_borda_count(a, b, weights=None) '''
        a = [2, 1, 0, 3, 4]
        b = [2, 1, 4, 0, 3]
        test_output = [3, 4, 0, 1, 2]
        output = self.br.simple_borda_count(a, b)
        self.assertEqual(output, test_output)
        a = [9, 2, 1, 0, 3, 4]
        output = self.br.simple_borda_count(a, b)
        self.assertEqual(output, test_output)
        output = self.br.simple_borda_count(a, b, weights=[2.0, 2.0])
        self.assertEqual(output, test_output)
        output = self.br.simple_borda_count(a, b, weights=[1.0, 2.0])
        test_output = [3, 0, 4, 1, 2]
        self.assertEqual(output, test_output)
        a = [90, 15, 65, 45, 78, 71, 69, 94, 57, 18, 92, 60, 14, 79, 70, 59, 7,
             85, 54, 89, 86, 66, 72, 80, 9, 93, 63, 46, 30, 8, 44, 3, 58,
             13, 67, 42, 84, 29, 40, 6, 77, 11, 61, 39, 81, 34, 88, 38, 28,
             43, 62, 47, 22, 5, 97, 95, 96, 32, 64, 91, 20, 53, 27, 21, 73,
             33, 16, 49, 24, 19, 68, 83, 17, 2, 41, 55, 51, 56, 23, 82, 31,
             87, 25, 37, 50, 35, 36, 76, 75, 10, 74, 12, 4, 1, 0, 52, 26, 48]
        b = [44, 34, 53, 38, 86, 18, 96, 97, 95, 63, 94, 32, 24, 70, 93, 31,
             85, 90, 69, 15, 50, 9, 68, 21, 67, 62, 76, 23, 42, 78, 87, 5, 71,
             22, 30, 4, 45, 79, 58, 89, 39, 88, 12, 80, 3, 6, 84, 83, 81, 74,
             60, 57, 73, 16, 13, 77, 28, 82, 20, 33, 25, 64, 40, 75, 91, 36,
             72, 19, 61, 11, 26, 17, 92, 66, 59, 52, 7, 56, 37, 47, 41, 49, 54,
             65, 0, 35, 55, 8, 43, 27, 51, 2, 10, 48, 14, 46, 1, 29]
        test_output = [48, 1, 10, 0, 35, 52, 26, 51, 2, 37, 55, 41, 56, 27, 36,
                       75, 49, 17, 25, 74, 43, 19, 82, 29, 12, 47, 4, 33, 91,
                       46, 16, 64, 20, 83, 8, 73, 76, 87, 11, 61, 14, 23, 28,
                       50, 40, 54, 31, 77, 66, 7, 68, 81, 59, 72, 13, 88, 21,
                       22, 65, 5, 6, 39, 84, 92, 24, 3, 62, 58, 32, 80, 42, 53,
                       95, 30, 96, 60, 97, 57, 67, 89, 38, 79, 34, 9, 45, 93,
                       71, 63, 78, 85, 44, 70, 69, 86, 15, 90, 94, 18]
        output = self.br.simple_borda_count(a, b)
        self.assertEqual(output, test_output)

class TestPICOVizRobot(unittest.TestCase):
    
    pv = PICOVizRobot()
    ex_path = os.path.dirname(__file__) + "/ex/"
    
    def test_init(self):
        pvr = PICOVizRobot()
        elements = ["population", "intervention", "outcomes"]
        self.assertEqual(pvr.elements, elements)
        for e in elements:
            self.assertTrue(e in pvr.PCA_dict)
        
    def test_postprocess_embedding(self):
        before = np.load(self.ex_path + "before.npy")
        after = np.load(self.ex_path + "after.npy")
        test = self.pv.postprocess_embedding(before)
        self.assertEqual(np.array_equal(after, test), True)
        
    def test_tokenize(self):
        with open(self.ex_path + "pico_viz.json") as datafile:
            data = json.load(datafile)
        test = data["token_start"]
        tok = self.pv.tokenize(test)
        end = data["token_end"]
        self.assertEqual(tok, end)
        
    def test_annotate(self):
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
