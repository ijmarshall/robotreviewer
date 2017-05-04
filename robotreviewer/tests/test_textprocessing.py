import json
import os
import unittest

import robotreviewer.textprocessing.tokenizer as t
from robotreviewer.textprocessing.abbreviations import Abbreviations
from robotreviewer.textprocessing.pdfreader import PdfReader


class TestAbbreviations(unittest.TestCase):

    test_sentences = [
        "Long-term androgen suppression plus radiotherapy (AS+RT) is standard treatment of high-risk prostate cancer.",
        "To compare the test-retest reliability, convergent validity, and overall feasibility/ usability of activity-based (AB) and time-based (TB) approaches for obtaining self-reported moderate-to-vigorous physical activity (MVPA) from adolescents.",
        "This study was conducted to determine if prophylactic cranial irradiation (PCI) improves survival in locally advanced non-small-cell lung cancer (LA-NSCLC)",
        "Alternatives to cytotoxic agents are desirable for patients with HIV-associated Kaposi's sarcoma (KS).",
        "The primary objective was assessment of antitumor activity using modified AIDS Clinical Trial Group (ACTG) criteria for HIV-KS.",
        "To determine the effectiveness of bortezomib plus irinotecan and bortezomib alone in patients with advanced gastroesophageal junction (GEJ) and gastric adenocarcinoma."
    ]

    num_abbrevs = [1, 3, 2, 1, 1, 1]

    def test_init(self):
        ''' test for Abbreviations.__init__() '''
        for i in range(len(self.test_sentences)):
            a = Abbreviations(self.test_sentences[i])
            for key in a.dictionary:
                self.assertEqual(key in self.test_sentences[i], True)
        for i in range(len(self.test_sentences)):
            a = Abbreviations(self.test_sentences[i])
            self.assertEqual(len(a.dictionary), self.num_abbrevs[i])

class TestPdfReader(unittest.TestCase):
    
    pdf = PdfReader()
    ex_path = os.path.dirname(__file__) + "/ex/"
    
    def test_convert(self):
        ''' test for PdfReader.convert(pdf_binary) '''
        with open(self.ex_path + "pdf_as_list.txt") as infile:
            lst = json.loads(infile.read())
        pdf_binary = bytes(lst)
        out = self.pdf.convert(pdf_binary)
        grob = out.data["grobid"]
        with open(self.ex_path + "pdffile.json") as datafile:
            test = json.load(datafile)
        self.assertEqual(grob, test)
        
    def test_run_grobid(self):
        ''' test for PdfReader.run_grobid(pdf_binary) '''
        with open(self.ex_path + "pdf_as_list.txt") as infile:
            lst = json.loads(infile.read())
        pdf_binary = bytes(lst)
        xml = self.pdf.run_grobid(pdf_binary)
        with open(self.ex_path + "run_grobid.txt") as infile:
            xmltest = infile.read()
        # can't compare xml to xmltest as they contain the date they were
        #  generated, so would be equal in all but that
        # parsing the xml removes that date information but leaves the rest
        parsed = self.pdf.parse_xml(xml)
        parsedtest = self.pdf.parse_xml(xmltest)
        self.assertEqual(parsed.data, parsedtest.data)
        
    def test_parse_xml(self):
        ''' test for PdfReader.parse_xml(xml_string) '''
        with open(self.ex_path + "run_grobid.txt") as infile:
            xml = infile.read()
        grob = self.pdf.parse_xml(xml)
        grob = grob.data["grobid"]
        with open(self.ex_path + "pdffile.json") as datafile:
            test = json.load(datafile)
        self.assertEqual(grob, test)

class TestTokenizer(unittest.TestCase):

    def test_spacy(self):
        ''' test that tokenizer loads correctly '''
        self.assertTrue(t.nlp is not None)
