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
#           Byron Wallce <byron.wallace@northeastern.edu>

import json
import uuid
import os
import robotreviewer
from sklearn.feature_extraction.text import HashingVectorizer
import pickle
import numpy as np
import re
import scipy
from scipy.sparse import hstack


class BiasAbRobot:

    def __init__(self):
        
        with open(robotreviewer.get_data(os.path.join('bias_ab', 'bias_prob_clf.pck')), 'rb') as f:
            self.clf = pickle.load(f)

        self.vec = HashingVectorizer(ngram_range=(1, 3), stop_words='english')


    def api_annotate(self, articles):

        """
        Annotate abstract of clinical trial report
            
        """


        if not all(('ab' in article) and ('ti' in article) for article in articles):
            raise Exception('Abstract bias model requires titles and abstracts to be able to complete annotation')
        
        X = self.vec.transform([r['ti'] + '\n\n' + r['ab'] for r in articles])

        probs = self.clf.predict_proba(X)[:,1].tolist()
        out = []
        for i in probs:
            
            row = {"prob_low_rob": i}
            
            out.append(row)
        return out
