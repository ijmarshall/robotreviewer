"""
the PICORobot class takes the title and abstract of a clinical trial as
input, and returns Population, Comparator/Intervention Outcome
information in the same format, which can easily be converted to JSON.

The model was derived using the "Supervised Distant Supervision" strategy
introduced in our paper "Extracting PICO Sentences from Clinical Trial Reports
using Supervised Distant Supervision".
"""



import uuid
import logging

import numpy as np
import os

import robotreviewer

from robotreviewer.ml.ner_data_utils import CoNLLDataset
from robotreviewer.ml.ner_model import NERModel
from robotreviewer.ml.ner_config import Config
import robotreviewer
from itertools import chain
from robotreviewer.textprocessing import tokenizer

log = logging.getLogger(__name__)


class PICOSpanRobot:

    def __init__(self):
        """


        """
        logging.debug("Loading PICO LSTM-CRF")
        config = Config()
        # build model
        self.model = NERModel(config)
        self.model.build()
        self.model.restore_session(os.path.join(robotreviewer.DATA_ROOT, "pico_spans/model.weights/"))
        # self.model.restore_session("/home/iain/Code/robotlabs/pico_lstm/EBM-NLP/models/lstm-crf/results/test/model.weights/")
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
            ti = data['parsed_text'][:TI_LEN]
            ab = data['parsed_text'][:AB_LEN]
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
        
        for sent in chain(article['title'].sents, article['abstract'].sents):
            words = [w.text for w in sent]
            preds = self.model.predict(words)
            
            last_label = "N"
            span = []
            
            for w, p in zip(words, preds):
                
                if p != last_label and span:
                    out[label_dict[last_label]].append(' '.join(span).strip())
                    span = []
                    
                if p != "N":
                    span.append(w)

                last_label = p

            if last_label != "N":
                out[label_dict[last_label]].append(' '.join(span).strip())

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
