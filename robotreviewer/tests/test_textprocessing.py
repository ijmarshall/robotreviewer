import json
import unittest

import robotreviewer.textprocessing.tokenizer as t
from robotreviewer.textprocessing.abbreviations import Abbreviations


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

    def test_abbreviations(self):
        for i in range(len(self.test_sentences)):
            a = Abbreviations(self.test_sentences[i])
            for key in a.dictionary:
                self.assertEqual(key in self.test_sentences[i], True)

    def test_num_abbreviations(self):
        for i in range(len(self.test_sentences)):
            a = Abbreviations(self.test_sentences[i])
            self.assertEqual(len(a.dictionary), self.num_abbrevs[i])

class TestTokenizer(unittest.TestCase):

    def test_spacy(self):
        self.assertTrue(t.nlp is not None)
