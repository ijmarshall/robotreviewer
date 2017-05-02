import json
import unittest
import os
import numpy as np

import robotreviewer
from robotreviewer.data_structures import MultiDict
from robotreviewer.robots.rationale_robot import BiasRobot
from robotreviewer.robots.pico_robot import PICORobot
from robotreviewer.robots.pico_viz_robot import PICOVizRobot
from robotreviewer.robots.pubmed_robot import PubmedRobot
from robotreviewer.robots.rct_robot import RCTRobot

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

class TestPICORobot(unittest.TestCase):
        
    pr = PICORobot()
        
    def test_get_positional_features(self):
        ''' test for PICORobot._get_positional_features(sentences) '''
        before = ['understanding the full burden of disease among them has been challenging as direct estimates of Men who Have Sex with Men (MSM) numbers in the general population have been largely unavailable.', 'We describe the population of Men who Have Sex with Men (MSM) in New York City, compare their demographics, risk behaviours, and new HIV and primary and secondary (P&S) syphilis rates with those of men who have sex with women (MSW), and examine trends in disease rates among Men who Have Sex with Men (MSM).', 'Methods Population denominators and demographic and behav-ioural data were obtained from population-based behavioural surveys during 2005e2008.', 'Numbers of new HIV and P&S syphilis diagnoses were extracted from citywide disease surveillance registries.', 'We calculated overall, age-and race/ethnicity-specific case rates and rate ratios for Men who Have Sex with Men (MSM) and MSW, and analysed trends in Men who Have Sex with Men (MSM) rates by age and race/ethnicity.', 'Results The average prevalence of same-sex behaviour among sexually active men during 2005e2008 (5.0%; 95% CI 4.5 to 5.6) differed by age (peaking at 8% among 40e49-year-old men) and race/ethnicity (2.3% among non-Hispanic black men; 7.4% among non-Hispanic white men).', 'Compared to MSW, Men who Have Sex with Men (MSM) differed significantly on all demographics and reported a higher prevalence of condom use at last sex and of HIV testing, but also more sex partners; 38.4% of Men who Have Sex with Men (MSM) and 13.6% of MSW reported Âź3 partners in the last year (p<0.001).', 'Men who Have Sex with Men (MSM) HIV and P&S syphilis rates were 2526.9/100 000 and 707.0/100 000, each of which was over 140 times MSW rates.', 'Rates were highest among young and black Men who Have Sex with Men (MSM)', '(See Abstract LBO-1.5 table 1).', 'Over 4 years, HIV rates more than doubled and P&S syphilis rates increased sixfold among 18e29-year-old Men who Have Sex with Men (MSM) to reach 8870.0/100 000 and 2900.4/100 000 in 2008, respectively.', 'Conclusions', 'The substantial population of Men who Have Sex with Men (MSM) in NYC is at high risk for transmission of sexually transmitted infections given high disease rates and ongoing risk behaviours.', 'There is significant overlap between HIVand P&S syphilis epidemics in NYC with the relatively small subgroups of young and non-Hispanic black Men who Have Sex with Men (MSM) disproportionately affected.', 'Integration of HIV and STD case data would allow for better identification and characterisation of the population affected by these synergistic epidemics.', 'Intensified and innovative efforts to implement and evaluate prevention programs are required.\n\n\n', 'Late breaker poster session\n\n\nwith\n\n\nLBP1.\n\n\n\n\n\n\n\n']
        after = [{'DocumentPositionQuintile0': 1}, {'DocumentPositionQuintile0': 1}, {'DocumentPositionQuintile0': 1}, {'DocumentPositionQuintile0': 1}, {'DocumentPositionQuintile1': 1}, {'DocumentPositionQuintile1': 1}, {'DocumentPositionQuintile1': 1}, {'DocumentPositionQuintile2': 1}, {'DocumentPositionQuintile2': 1}, {'DocumentPositionQuintile2': 1}, {'DocumentPositionQuintile2': 1}, {'DocumentPositionQuintile3': 1}, {'DocumentPositionQuintile3': 1}, {'DocumentPositionQuintile3': 1}, {'DocumentPositionQuintile4': 1}, {'DocumentPositionQuintile4': 1}, {'DocumentPositionQuintile4': 1}]
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
    
    def test_init(self):
        ''' test for PICOVizRobot.__init__() '''
        '''
        pvr = PICOVizRobot()
        elements = ["population", "intervention", "outcomes"]
        self.assertEqual(pvr.elements, elements)
        for e in elements:
            self.assertTrue(e in pvr.PCA_dict)
        '''
        pass
        
    def test_postprocess_embedding(self):
        ''' test for PICOVizRobot.postprocess_embedding(H) '''
        before = np.load(self.ex_path + "before.npy")
        after = np.load(self.ex_path + "after.npy")
        test = self.pv.postprocess_embedding(before)
        self.assertTrue(np.array_equal(after, test))
        
    def test_tokenize(self):
        ''' test for PICOVizRobot.tokenize(text) '''
        with open(self.ex_path + "pico_viz.json") as datafile:
            data = json.load(datafile)
        test = data["token_start"]
        tok = self.pv.tokenize(test)
        end = data["token_end"]
        self.assertEqual(tok, end)
        
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
        
    def test_kv_transform(self):
        ''' test for KerasVectorizer.transform(raw_documents) '''
        with open(self.ex_path + "rct.json") as data:
            data = json.load(data)
        kv = self.rct.cnn_vectorizer
        raw_documents = data["raw_documents"]
        test = np.load(self.ex_path + "kv_transform.npy")
        self.assertTrue(np.array_equal(kv.transform(raw_documents), test))
