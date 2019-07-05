"""
the TripBiasRobot class takes the *abstract* of a clinical trial as
input as a string, and returns bias information as a dict which
can be easily converted to JSON.

    text = "Streptomycin Treatment of Pulmonary Tuberculosis: A Medical Research Council Investigation..."

    robot = BiasRobot()
    annotations = robot.annotate(text)

Based on the document-level models which were validated in the paper:

Marshall IJ, Kuiper J, & Wallace BC. RobotReviewer: evaluation of a system for automatically assessing bias in clinical trials. Journal of the American Medical Informatics Association 2015.doi:10.1093/jamia/ocv044

Returns only data about:
- Random sequence generation
- Allocation concealment
- Blinding
as other domains did not make as much sense to detect from abstact data
"""

# Authors:  Iain Marshall <mail@ijmarshall.com>
#           Joel Kuiper <me@joelkuiper.com>
#           Byron Wallce <byron.wallace@utexas.edu>

import json
import uuid
import os


import robotreviewer
from robotreviewer.ml.classifier import MiniClassifier
from robotreviewer.ml.vectorizer import ModularVectorizer

import numpy as np
import re

class BiasAbRobot:

    def __init__(self, top_k=3):
        """
        `top_k` refers to 'top-k recall'.

        top-1 recall will return the single most relevant sentence
        in the document, and top-3 recall the 3 most relevant.

        The validation study assessed the accuracy of top-3 and top-1
        and we suggest top-3 as default
        """


        self.doc_clf = MiniClassifier(robotreviewer.get_data(os.path.join('bias_ab', 'bias_ab.npz')))
        self.vec = ModularVectorizer(norm=None, non_negative=True, binary=True, ngram_range=(1, 2))
        self.bias_domains = ['random_sequence_generation', 'allocation_concealment', 'blinding_participants_personnel']
        self.top_k = top_k



    def api_annotate(self, articles, top_k=None):

        """
        Annotate full text of clinical trial report
        `top_k` can be overridden here, else defaults to the class
        default set in __init__
        """
        if not all(('ab' in article) and ('ti' in article) for article in articles):
            raise Exception('Abstract bias model requires titles and abstracts to be able to complete annotation')


        if top_k is None:
            top_k = self.top_k



        out = []

        for article in articles:
            doc_text = article['ti'] + "  " + article['ab']
            row = {}
            for domain in self.bias_domains:

                #
                # build up document feature set
                #
                self.vec.builder_clear()

                # uni-bigrams
                self.vec.builder_add_docs([doc_text])

                # uni-bigrams/domain interaction
                self.vec.builder_add_docs([(doc_text, domain)])
                x = self.vec.builder_transform()
                bias_pred = self.doc_clf.predict(x)
                bias_class = ["high/unclear", "low"][bias_pred[0]]
                row[domain] = {"judgement": bias_class}
            out.append(row)
        return out 
