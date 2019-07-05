"""
the PICORobot class takes the title and abstract of a clinical trial as
input, and returns Population, Comparator/Intervention Outcome
information in the same format, which can easily be converted to JSON.

The model was described using the corpus and methods reported in our
ACL 2018 paper "A corpus with multi-level annotations of patients, 
interventions and outcomes to support language processing for medical 
literature": https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6174533/.
"""


import string 

import uuid
import logging

import numpy as np
import os
import spacy 

import robotreviewer

from robotreviewer.ml.ner_data_utils import CoNLLDataset
from robotreviewer.ml.ner_model import NERModel
from robotreviewer.ml.ner_config import Config
import robotreviewer
from itertools import chain
from robotreviewer.textprocessing import tokenizer

log = logging.getLogger(__name__)

from celery.contrib import rdb


def cleanup(spans):
    '''
    A helper (static) function for prettifying / deduplicating
    the PICO snippets extracted by the model.
    '''
    def clean_span(s):
        s_clean = s.strip()
        # remove punctuation
        s.strip(string.punctuation)

        # remove 'Background:' when we pick it up
        s_clean = s_clean.replace("Background", "")
        return s_clean

    cleaned_spans = [clean_span(s) for s in spans]
    # dedupe
    return list(set(cleaned_spans))


class PICOSpanRobot:

    def __init__(self):
        """
        This bot tags sequences of words from abstracts as describing
        P,I, or O elements. 
        """
        logging.debug("Loading PICO LSTM-CRF")
        config = Config()
        # build model
        self.model = NERModel(config)
        self.model.build()

        self.model.restore_session(os.path.join(robotreviewer.DATA_ROOT, "pico_spans/model.weights/"))
        logging.debug("PICO classifiers loaded")


    def api_annotate(self, articles):

        if not (all(('parsed_ab' in article for article in articles)) and all(('parsed_ti' in article for article in articles))):
            raise Exception('PICO span model requires a title and abstract to be able to complete annotation')

        
        annotations = []
        for article in articles:
            if article.get('skip_annotation'):
                annotations.append([])
            else:
                annotations.append(self.annotate({"title": article['parsed_ti'], "abstract": article['parsed_ab']}))
               
        return annotations


    def pdf_annotate(self, data):
        if data.get("abstract") is not None and data.get("title") is not None:
            ti = tokenizer.nlp(data["title"])
            ab = tokenizer.nlp(data["abstract"])
        elif data.get("parsed_text") is not None:
            # then just use the start of the document
            TI_LEN = 30
            AB_LEN = 500
            # best guesses based on sample of RCT abstracts + aiming for 95% centile
            ti = tokenizer.nlp(data['parsed_text'][:TI_LEN].string)
            ab = tokenizer.nlp(data['parsed_text'][:AB_LEN].string)
        else:
            # else can't proceed
            return data


        data.ml["pico_span"] = self.annotate({"title": ti, "abstract": ab})

        return data


    def annotate(self, article):

        """
        Annotate abstract of clinical trial report
        """

        label_dict = {"1_p": "population", "1_i": "interventions", "1_o": "outcomes"}
    
        out = {"population": [],
               "interventions": [],
               "outcomes": []}
        
        '''
        rdb.set_trace()
        if type(article['abstract']) == spacy.tokens.span.Span:

            article_sentences = [article['abstract']]
        else: 
            article_sentences = article['abstract'].sents
        '''
        for sent in chain(article['title'].sents, article['abstract'].sents):
            words = [w.text for w in sent]
            preds = self.model.predict(words)
           
            last_label = "N"
            start_idx = 0
            
            for i, p in enumerate(preds):
                
                if p != last_label and last_label != "N":
                    out[label_dict[last_label]].append(sent[start_idx: i].text.strip())
                    start_idx = i

                if p != last_label and last_label == "N":
                    start_idx = i

                last_label = p

            if last_label != "N":
                out[label_dict[last_label]].append(sent[start_idx:].text.strip())


        for e in out: 
            out[e] = cleanup(out[e])

        return out


    @staticmethod
    def get_marginalia(data):
        """
        Get marginalia formatted for Spa from structured data
        """
        marginalia = [{"type": "PICO text from abstracts",
                      "title": "PICO characteristics",
                      "annotations": [],
                      "description":  data["ml"]["pico_span"]}]
        return marginalia
        

def main():
    pass


if __name__ == '__main__':
    main()
