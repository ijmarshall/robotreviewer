"""
the BiasAbRobot class takes the *abstract* of a clinical trial as
input as a string, and returns bias information as a dict which
can be easily converted to JSON.

V2.0

Returns an indicative probability that the article is at low risk of bias, based on the abstract alone.

Trained on 



"""

# Authors:  Iain Marshall <mail@ijmarshall.com>
#           Joel Kuiper <me@joelkuiper.com>
#           Byron Wallce <byron.wallace@utexas.edu>

import json
import uuid
import os
import robotreviewer
from sklearn.feature_extraction.text import HashingVectorizer
import pickle
import numpy as np
import re
from scipy.sparse import hstack


class BiasAbRobot:

    def __init__(self, top_k=3):
        """
        `top_k` refers to 'top-k recall'.

        top-1 recall will return the single most relevant sentence
        in the document, and top-3 recall the 3 most relevant.

        The validation study assessed the accuracy of top-3 and top-1
        and we suggest top-3 as default
        """

        with open(robotreviewer.get_data(os.path.join('bias_ab', 'domain_clf.pck')), 'rb') as f:
            self.domain_clf = pickle.load(f)

        with open(robotreviewer.get_data(os.path.join('bias_ab', 'overall_clf.pck')), 'rb') as f:
            self.overall_clf = pickle.load(f)

        self.vec = HashingVectorizer(ngram_range=(1, 2))


    def api_annotate(self, articles, top_k=None):

        """
        Annotate full text of clinical trial report
        `top_k` can be overridden here, else defaults to the class
        default set in __init__
        """


        if not all(('ab' in article) and ('ti' in article) for article in articles):
            raise Exception('Abstract bias model requires titles and abstracts to be able to complete annotation')


        X_domains_t = [[], [], [], [], []]

        for article in articles:
            for i in range(4):
                for j in range(4):
                    if i==j:
                        X_domains_t[i].append(article['ti'] + '\n\n' + article['ab'])
                    else:
                        X_domains_t[i].append("")
                X_domains_t[4].append(article['ti'] + '\n\n' + article['ab'])

        X_vecs = []

        for xdt_i in X_domains_t:
            X_vecs.append(self.vec.transform(xdt_i))

        X_domains = hstack(X_vecs)
        X_domains = X_domains.tocsr()

        prob_domains = self.domain_clf.predict_proba(X_domains)[:,1]
        X_all = prob_domains.reshape((int(prob_domains.shape[0]/4), 4))
        prob_all = self.overall_clf.predict_proba(X_all)[:,1].tolist()

        out = []

        for i in prob_all:
            
            row = {"prob_low_rob": i}
            
            out.append(row)
        return out
