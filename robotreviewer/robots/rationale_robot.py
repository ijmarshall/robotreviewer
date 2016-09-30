"""
the BiasRobot class takes the full text of a clinical trial as
input as a string, and returns bias information as a dict which
can be easily converted to JSON.

    text = "Streptomycin Treatment of Pulmonary Tuberculosis: A Medical Research Council Investigation..."

    robot = BiasRobot()
    annotations = robot.annotate(text)

Implements the models which were validated in the paper:

Marshall IJ, Kuiper J, & Wallace BC. RobotReviewer: evaluation of a system for automatically assessing bias in clinical trials. Journal of the American Medical Informatics Association 2015.doi:10.1093/jamia/ocv044
"""

# Authors:  Iain Marshall <mail@ijmarshall.com>
#           Joel Kuiper <me@joelkuiper.com>
#           Byron Wallce <byron.wallace@utexas.edu>

import json
import uuid
import os
import pickle

import robotreviewer
from robotreviewer.textprocessing import tokenizer
from robotreviewer.ml.classifier import MiniClassifier
from robotreviewer.ml.vectorizer import ModularVectorizer

import sys
sys.path.append('robotreviewer/ml') # need this for loading the rationale_CNN module
from rationale_CNN import RationaleCNN, Document

import numpy as np
import re

import keras
import keras.backend as K
from keras.models import load_model


class BiasRobot:

    def __init__(self, top_k=3):
        """
        `top_k` refers to 'top-k recall'.

        top-1 recall will return the single most relevant sentence
        in the document, and top-3 recall the 3 most relevant.

        The validation study assessed the accuracy of top-3 and top-1
        and we suggest top-3 as default

        """
        self.bias_domains = ['Random sequence generation']
        self.top_k = top_k

        # load in preprocessor
        preprocesser = pickle.load(open("robotreviewer/data/keras/vectorizers/preprocessor.pickle", 'rb'))

        # and now load in the model; assumes h5 file bundling
        # architecture and params naturally
        model_arch_file    = "robotreviewer/data/keras/models/rationale-CNN_model.json"
        model_weights_file = "robotreviewer/data/keras/models/rationale-CNN_RSG.hdf5"
        self.model = RationaleCNN(preprocesser,
                                  document_model_architecture_path=model_arch_file,
                                  document_model_weights_path=model_weights_file)   

    def annotate(self, data, top_k=None):
        """
        Annotate full text of clinical trial report
        `top_k` can be overridden here, else defaults to the class
        default set in __init__

        """
        top_k = self.top_k if not top_k else top_k
        
        doc_text = data.get('parsed_text')
        if not doc_text:
            return data # we've got to know the text at least..

        doc_len = len(data['text'])
        doc_sents = [sent.string for sent in doc_text.sents]#[:self.max_doclen] # cap maximum number of sentences
        # doc_sents = [str(''.join(c for c in sent if ord(c) < 128)) for sent in doc_sents] # delete non-ascii characters
        doc_sent_start_i = [sent.start_char for sent in doc_text.sents]
        doc_sent_end_i = [sent.end_char for sent in doc_text.sents]

        structured_data = []
        for domain in self.bias_domains:
            # vectorize document
            doc = Document(None, doc_sents)
            bias_prob, high_prob_sent_indices = self.model.predict_and_rank_sentences_for_doc(doc, num_rationales=top_k)
            bias_pred = int(bias_prob > .5)

            high_prob_sents = [doc_sents[i] for i in high_prob_sent_indices]
            high_prob_start_i = [doc_sent_start_i[i] for i in high_prob_sent_indices]
            high_prob_end_i = [doc_sent_end_i[i] for i in high_prob_sent_indices]
            high_prob_prefixes = [doc_text.string[max(0, offset-20):offset] for offset in high_prob_start_i]
            high_prob_suffixes = [doc_text.string[offset: min(doc_len, offset+20)] for offset in high_prob_end_i]
            high_prob_sents_j = " ".join(high_prob_sents)
            sent_domain_interaction = "-s-" + domain

            bias_class = ["high/unclear", "low"][bias_pred]
            annotation_metadata = [{"content": sent[0],
                                    "position": sent[1],
                                    "uuid": str(uuid.uuid1()),
                                    "prefix": sent[2],
                                    "suffix": sent[3]} for sent in zip(high_prob_sents, high_prob_start_i,
                                       high_prob_prefixes,
                                       high_prob_suffixes)]

            structured_data.append({
                "domain": domain,
                "judgement": bias_class,
                "annotations": annotation_metadata})
        data.ml["bias"] = structured_data
        return data

    @staticmethod
    def get_marginalia(data):
        """
        Get marginalia formatted for Spa from structured data
        """
        marginalia = []

        for row in data['bias']:
            marginalia.append({
                        "type": "Risk of Bias",
                        "title": row['domain'],
                        "annotations": row['annotations'],
                        "description": "**Overall risk of bias prediction**: {}".format(row['judgement'])
                        })
        return marginalia

    @staticmethod
    def get_domains():
        return [u'Random sequence generation',
                u'Allocation concealment',
                u'Blinding of participants and personnel',
                u'Blinding of outcome assessment',
                u'Incomplete outcome data',
                u'Selective reporting']
