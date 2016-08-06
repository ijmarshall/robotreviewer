"""
the Randomized Control Trial (RCT) robot predicts whether a given
*abstract* (not full-text) describes an RCT.

    title =    '''Does usage of a parachute in contrast to free fall prevent major trauma?: a prospective randomised-controlled trial in rag dolls.'''
    abstract = '''PURPOSE: It is undisputed for more than 200 years that the use of a parachute prevents major trauma when falling from a great height. Nevertheless up to date no prospective randomised controlled trial has proven the superiority in preventing trauma when falling from a great height instead of a free fall. The aim of this prospective randomised controlled trial was to prove the effectiveness of a parachute when falling from great height. METHODS: In this prospective randomised-controlled trial a commercially acquirable rag doll was prepared for the purposes of the study design as in accordance to the Declaration of Helsinki, the participation of human beings in this trial was impossible. Twenty-five falls were performed with a parachute compatible to the height and weight of the doll. In the control group, another 25 falls were realised without a parachute. The main outcome measures were the rate of head injury; cervical, thoracic, lumbar, and pelvic fractures; and pneumothoraxes, hepatic, spleen, and bladder injuries in the control and parachute groups. An interdisciplinary team consisting of a specialised trauma surgeon, two neurosurgeons, and a coroner examined the rag doll for injuries. Additionally, whole-body computed tomography scans were performed to identify the injuries. RESULTS: All 50 falls-25 with the use of a parachute, 25 without a parachute-were successfully performed. Head injuries (right hemisphere p = 0.008, left hemisphere p = 0.004), cervical trauma (p < 0.001), thoracic trauma (p < 0.001), lumbar trauma (p < 0.001), pelvic trauma (p < 0.001), and hepatic, spleen, and bladder injures (p < 0.001) occurred more often in the control group. Only the pneumothoraxes showed no statistically significant difference between the control and parachute groups. CONCLUSIONS: A parachute is an effective tool to prevent major trauma when falling from a great height.'''

    rct_robot = RCTRobot()
    annotations = rct_robot.annotate(title, abstract)

This model was trained on the Cochrane crowd dataset. 
"""

# Authors:  Iain Marshall <mail@ijmarshall.com>
#           Joel Kuiper <me@joelkuiper.com>
#           Byron Wallce <byron.wallace@utexas.edu>

import json
import uuid
import os

import pickle 

import robotreviewer
from robotreviewer.ml.classifier import MiniClassifier
from sklearn.feature_extraction.text import HashingVectorizer

import numpy as np
import re

# using PubMed's stoplist, not NLTK's, to be consistent with training
stopwords = ["a","about","above","abs","accordingly","across","after","afterwards","again","against","all","almost","alone","along","already","also","although","always","am","among","amongst","an","analyze","and","another","any","anyhow","anyone","anything","anywhere","applicable","apply","are","arise","around","as","assume","at","be","became","because","become","becomes","becoming","been","before","beforehand","being","below","beside","besides","between","beyond","both","but","by","came","can","cannot","cc","cm","come","compare","could","de","dealing","department","depend","did","discover","dl","do","does","done","due","during","each","ec","ed","effected","eg","either","else","elsewhere","enough","especially","et","etc","ever","every","everyone","everything","everywhere","except","find","for","found","from","further","gave","get","give","go","gone","got","gov","had","has","have","having","he","hence","her","here","hereafter","hereby","herein","hereupon","hers","herself","him","himself","his","how","however","hr","i","ie","if","ii","iii","immediately","importance","important","in","inc","incl","indeed","into","investigate","is","it","its","itself","just","keep","kept","kg","km","last","latter","latterly","lb","ld","letter","like","ltd","made","mainly","make","many","may","me","meanwhile","mg","might","ml","mm","mo","more","moreover","most","mostly","mr","much","mug","must","my","myself","namely","nearly","necessarily","neither","never","nevertheless","next","no","nobody","noone","nor","normally","nos","noted","nothing","now","nowhere","obtained","of","off","often","on","only","onto","other","others","otherwise","ought","our","ours","ourselves","out","over","overall","owing","own","oz","particularly","per","perhaps","pm","precede","predominantly","present","presently","previously","primarily","promptly","pt","quickly","quite","rather","readily","really","recently","refs","regarding","relate","said","same","seem","seemed","seeming","seems","seen","seriously","several","shall","she","should","show","showed","shown","shown","shows","significantly","since","slightly","so","some","somehow","someone","something","sometime","sometimes","somewhat","somewhere","soon","specifically","still","strongly","studied","sub","substantially","such","sufficiently","take","tell","th","than","that","the","their","theirs","them","themselves","then","thence","there","thereafter","thereby","therefore","therein","thereupon","these","they","this","thorough","those","though","through","throughout","thru","thus","to","together","too","toward","towards","try","type","ug","under","unless","until","up","upon","us","use","used","usefully","usefulness","using","usually","various","very","via","was","we","were","what","whatever","when","whence","whenever","where","whereafter","whereas","whereby","wherein","whereupon","wherever","whether","which","while","whither","who","whoever","whom","whose","why","will","with","within","without","wk","would","wt","yet","you","your","yours","yourself","yourselves","yr"]

class RCTRobot:

    def __init__(self):
        self.clf = MiniClassifier(robotreviewer.get_data('rct/rct.npz'))
        self.vectorizer = HashingVectorizer(binary=True, ngram_range=(1, 3), stop_words='english')

    def annotate(self, data):

        title_text = data['title']
        abstract_text = data['abstract']

        if title_text is None or abstract_text is None:
            # not much point in continuing unless we have a title and abstract
            return data

        merged_words = []
        # special indicators for titles
        merged_words = [u"TI_{0}".format(t) for t in title_text.split(" ") 
                        if not t in stopwords]
        merged_words.extend(abstract_text.split(" "))

        x = self.vectorizer.transform([" ".join(merged_words)])

        p_hat = self.clf.predict_proba(x)[0,0]
        y_hat = 1 if p_hat >= .5 else 0

        is_rct_str = "Not an RCT"
        if y_hat > 0: 
            is_rct_str = "RCT"

        marginalia = {"type": "Trial Design",
                      "title": "Is an RCT?",
                      "annotations": [],
                      "description":  "{0} (p={1:0.2f})".format(is_rct_str, p_hat)}


        structured_data = {"is_rct": bool(p_hat >= 0.5),
                           "prob_rct": p_hat}
        data.ml["rct"] = {"structured": structured_data,
                        "marginalia": marginalia}

        return data





