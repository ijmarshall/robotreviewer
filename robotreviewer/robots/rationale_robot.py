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
sys.path.append('robotreviewer/ml') # need this for loading the pickled SequenceVectorizer

import numpy as np
import re

import keras
import keras.backend as K


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

        # model
        self.model = keras.models.load_model('robotreviewer/data/keras/models/rationale.h5')

        # vectorizer
        self.vectorizer = pickle.load(open('robotreviewer/data/keras/vectorizers/rationale.p'))
        self.max_sentlen = self.vectorizer.maxlen
        self.max_doclen = self.model.layers[1].input_length / self.max_sentlen # hacky way to get out `max_doclen`

        # keras function for computing sentence weights and document probs
        inputs = self.model.inputs + [K.learning_phase()]
        outputs = [self.model.get_layer('sent_weights').output, self.model.get_layer('doc_probs').output]
        self.eval_doc = K.function(inputs, outputs)

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
        doc_sents = [sent.string for sent in doc_text.sents][:self.max_doclen] # cap maximum number of sentences
        doc_sents = [str(''.join(c for c in sent if ord(c) < 128)) for sent in doc_sents] # delete non-ascii characters
        doc_sent_start_i = [sent.start_char for sent in doc_text.sents]
        doc_sent_end_i = [sent.end_char for sent in doc_text.sents]

        structured_data = []
        for domain in self.bias_domains:
            # vectorize document
            X_seq = self.vectorizer.texts_to_sequences(doc_sents)
            nb_sentence, _ = X_seq.shape
            X_doc = np.zeros([self.max_doclen, self.max_sentlen])
            X_doc[self.max_doclen-min(self.max_doclen, nb_sentence):] = X_seq[:self.max_doclen]
            X = np.zeros([1, self.max_sentlen*self.max_doclen], dtype=np.int)
            X[0] = X_doc.reshape([self.max_sentlen*self.max_doclen])

            result = self.eval_doc([X, 1])
            doc_sents_preds, bias_probs = [output[0] for output in result]
            bias_pred = np.argmax(bias_probs)

            high_prob_sent_indices = np.argsort(doc_sents_preds)[:-top_k-1:-1] # top k, with no 1 first
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
