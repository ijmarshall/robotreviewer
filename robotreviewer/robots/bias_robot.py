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
#           Byron Wallace <byron@ccs.neu.edu>

import uuid

import numpy as np

import robotreviewer
from robotreviewer.ml.classifier import MiniClassifier
from robotreviewer.ml.vectorizer import ModularVectorizer

class BiasRobot:

    def __init__(self, top_k=3):
        """
        `top_k` refers to 'top-k recall'.

        top-1 recall will return the single most relevant sentence
        in the document, and top-3 recall the 3 most relevant.

        The validation study assessed the accuracy of top-3 and top-1
        and we suggest top-3 as default
        """


        self.sent_clf = MiniClassifier(robotreviewer.get_data('bias/bias_sent_level.npz'))
        self.doc_clf = MiniClassifier(robotreviewer.get_data('bias/bias_doc_level.npz'))

        self.vec = ModularVectorizer(norm=None, non_negative=True, binary=True, ngram_range=(1, 2), n_features=2**26)

        self.bias_domains = ['Random sequence generation', 'Allocation concealment', 'Blinding of participants and personnel', 'Blinding of outcome assessment', 'Incomplete outcome data', 'Selective reporting']

        self.top_k = top_k

    def pdf_annotate(self, data, top_k=None):

        """
        Annotate full text of clinical trial report
        `top_k` can be overridden here, else defaults to the class
        default set in __init__
        """
        if top_k is None:
            top_k = self.top_k


        doc_text = data.get('parsed_text')

        if not doc_text:
            # we've got to know the text at least..
            return data

        doc_len = len(data['text'])



        marginalia = []

        doc_sents = [sent.text for sent in doc_text.sents]
        doc_sent_start_i = [sent.start_char for sent in doc_text.sents]
        doc_sent_end_i = [sent.end_char for sent in doc_text.sents]

        structured_data = []

        for domain in self.bias_domains:

            doc_domains = [domain] * len(doc_sents)
            doc_X_i = zip(doc_sents, doc_domains)

            #
            # build up sentence feature set
            #
            self.vec.builder_clear()

            # uni-bigrams
            self.vec.builder_add_docs(doc_sents)

            # uni-bigrams/domain interactions
            self.vec.builder_add_docs(doc_X_i)

            doc_sents_X = self.vec.builder_transform()
            doc_sents_preds = self.sent_clf.decision_function(doc_sents_X)

            high_prob_sent_indices = np.argsort(doc_sents_preds)[:-top_k-1:-1] # top k, with no 1 first
            high_prob_sents = [doc_sents[i] for i in high_prob_sent_indices]
            high_prob_start_i = [doc_sent_start_i[i] for i in high_prob_sent_indices]
            high_prob_end_i = [doc_sent_end_i[i] for i in high_prob_sent_indices]
            high_prob_prefixes = [doc_text.text[max(0, offset-20):offset] for offset in high_prob_start_i]
            high_prob_suffixes = [doc_text.text[offset: min(doc_len, offset+20)] for offset in high_prob_end_i]
            high_prob_sents_j = " ".join(high_prob_sents)
            sent_domain_interaction = "-s-" + domain

            #
            # build up document feature set
            #
            self.vec.builder_clear()

            # uni-bigrams
            self.vec.builder_add_docs([doc_text.text])

            # uni-bigrams/domain interaction
            self.vec.builder_add_docs([(doc_text.text, domain)])

            # uni-bigrams/relevance interaction
            self.vec.builder_add_docs([(high_prob_sents_j, sent_domain_interaction)])

            X = self.vec.builder_transform()

            bias_pred = self.doc_clf.predict(X)
            bias_class = ["high/unclear", "low"][bias_pred[0]]
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
