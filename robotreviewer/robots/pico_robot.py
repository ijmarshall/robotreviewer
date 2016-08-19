"""
the PICORobot class takes the full text of a clinical trial as
input as a string, and returns Population, Comparator/Intervention
Outcome information as a dict which can be easily converted to JSON.

    text = "Streptomycin Treatment of Pulmonary Tuberculosis: A Medical Research Council Investigation..."

    robot = PICORobot()
    annotations = robot.annotate(text)

The model was derived using the "Supervised Distant Supervision" strategy
introduced in our paper "Extracting PICO Sentences from Clinical Trial Reports
using Supervised Distant Supervision".
"""

# Authors:  Iain Marshall <mail@ijmarshall.com>
#           Joel Kuiper <me@joelkuiper.com>
#           Byron Wallce <byron.wallace@utexas.edu>



import json
import uuid
import numpy as np
import sys
import os
import logging
from scipy.sparse import diags
import fnmatch
import re
import sklearn
from scipy.sparse import lil_matrix, csc_matrix
import numpy as np
import scipy as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import diags



import robotreviewer
from robotreviewer import config
from robotreviewer.ml.classifier import MiniClassifier
from robotreviewer.lexicons.drugbank import Drugbank
# from robotreviewer.textprocessing.abbreviations import Abbreviations
from robotreviewer.textprocessing import tokenizer


log = logging.getLogger(__name__)



class PICORobot:

    def __init__(self, top_k=2, min_k=1):
        """
        In most cases, a fixed number of sentences (top_k) will be
        returned for each document, *except* when the decision
        scores are below a threshold (i.e. the implication being
        that none of the sentences are relevant).

        top_k = the default number of sentences to retrive per
                document
        min_k = ensure that at at least min_k sentences are
                always returned


        """

        logging.debug("Loading PICO classifiers")

        self.P_clf = MiniClassifier(robotreviewer.get_data("pico/P_model.npz"))
        self.I_clf = MiniClassifier(robotreviewer.get_data("pico/I_model.npz"))
        self.O_clf = MiniClassifier(robotreviewer.get_data("pico/O_model.npz"))

        logging.debug("PICO classifiers loaded")

        logging.debug("Loading IDF weights")
        with open(robotreviewer.get_data("pico/P_idf.npz"), 'rb') as f:
            self.P_idf = diags(np.load(f, encoding='latin1').item().todense().A1, 0)

        with open(robotreviewer.get_data("pico/I_idf.npz"), 'rb') as f:
            self.I_idf = diags(np.load(f, encoding='latin1').item().todense().A1, 0)

        with open(robotreviewer.get_data("pico/O_idf.npz"), 'rb') as f:
            self.O_idf = diags(np.load(f, encoding='latin1').item().todense().A1, 0)

        logging.debug("IDF weights loaded")


        self.vec = PICO_vectorizer()
        self.models = [self.P_clf, self.I_clf, self.O_clf]
        self.idfs = [self.P_idf, self.I_idf, self.O_idf]
        self.PICO_domains = ["Population", "Intervention", "Outcomes"]


        # if config.USE_METAMAP:
        #     self.metamap = MetaMap.get_instance()

        self.top_k = top_k
        self.min_k = min_k


    def annotate(self, data, top_k=3, min_k=1, alpha=.7):

        """
        Annotate full text of clinical trial report
        `top_k` refers to 'top-k recall'.

        Default alpha was totally scientifically set.
        """


        doc_text = data.get("parsed_text")
        
        if not doc_text:
            # we've got to know the text at least..
            return data

        doc_len = len(data['text'])


        if top_k is None:
            top_k = self.top_k

        if min_k is None:
            min_k = self.min_k


        marginalia = []
        structured_data = []

        # abbr_resolver = Abbreviations(doc_text.text) # Not being used at the moment

        doc_sents = [sent.text for sent in doc_text.sents]
        doc_sent_start_i = [sent.start_char for sent in doc_text.sents]
        doc_sent_end_i = [sent.end_char for sent in doc_text.sents]

        # quintile indicators (w.r.t. document) for sentences
        positional_features = PICORobot._get_positional_features(doc_sents)

        for domain, model, idf in zip(self.PICO_domains, self.models, self.idfs):

            log.debug('Starting prediction')
            log.debug('vectorizing')
            doc_sents_X = self.vec.transform(doc_text, extra_features=positional_features, idf=idf)

            log.debug('predicting sentence probabilities')
            doc_sents_preds = model.predict_proba(doc_sents_X)

            log.info('finding best predictive sents')
            high_prob_sent_indices = np.argsort(doc_sents_preds)[:-top_k-1:-1]

            # filter
            filtered_high_prob_sent_indices = \
                                              high_prob_sent_indices[doc_sents_preds[high_prob_sent_indices] >= alpha]

            log.info('Prediction done!')

            if len(filtered_high_prob_sent_indices) < min_k:
                high_prob_sent_indices = high_prob_sent_indices[:min_k]
            else:
                high_prob_sent_indices = filtered_high_prob_sent_indices


            high_prob_sents = [doc_sents[i] for i in high_prob_sent_indices]
            high_prob_start_i = [doc_sent_start_i[i] for i in high_prob_sent_indices]
            high_prob_end_i = [doc_sent_end_i[i] for i in high_prob_sent_indices]
            high_prob_prefixes = [doc_text.string[max(0, offset-20):offset] for offset in high_prob_start_i]
            high_prob_suffixes = [doc_text.string[offset: min(doc_len, offset+20)] for offset in high_prob_end_i]

            annotations = [{"content": sent[0],
                            "position": sent[1],
                            "uuid": str(uuid.uuid1()),
                            "prefix": sent[2],
                            "suffix": sent[3]} for sent in zip(high_prob_sents, high_prob_start_i,
                                                               high_prob_prefixes,
                                                               high_prob_suffixes)]

            structured_data.append({"domain":domain,
                                    "text": high_prob_sents,
                                    "annotations": annotations})

        data.ml["pico_text"] = structured_data
        return data

    @staticmethod
    def _get_positional_features(sentences):
        ''' generate positional features here (quintiles) for doc sentences. '''
        num_sents = len(sentences)
        quintile_cutoff = num_sents / 5

        if quintile_cutoff == 0:
            sentence_quintiles = [{"DocTooSmallForQuintiles" : 1} for ii in range(num_sents)]
            log.warning("Tiny file encountered... len=%d" % num_sents)
        else:
            sentence_quintiles = [{"DocumentPositionQuintile%d" % (ii/quintile_cutoff): 1} for ii in range(num_sents)]
        return sentence_quintiles


    @staticmethod
    def get_marginalia(data):
        """
        Get marginalia formatted for Spa from structured data
        """
        marginalia = []
        for row in data['pico_text']:
            marginalia.append({
                "type": "PICO",
                "title": row['domain'],
                "annotations": row['annotations']
            })
        return marginalia

class PICO_vectorizer:

    def __init__(self):
        self.vectorizer = HashingVectorizer(ngram_range=(1, 2))
        self.dict_vectorizer = DictVectorizer()

        # These are set dynamically in training
        # but fixed here to match the end feature names
        # in the trained model. If the model is retrained then
        # these may have to change
        self.dict_vectorizer.feature_names_ = [
            'DocumentPositionQuintile0',
            'DocumentPositionQuintile1',
            'DocumentPositionQuintile2',
            'DocumentPositionQuintile3',
            'DocumentPositionQuintile4',
            'DocumentPositionQuintile5',
            'DocumentPositionQuintile6']
        self.dict_vectorizer.vocabulary_ = {k: i for i, k in enumerate(self.dict_vectorizer.feature_names_)}

        self.drugbank = Drugbank()

    def token_contains_number(self, token):
        return any(char.isdigit() for char in token)

    def is_number(self,num):
        try:
            float(num)
            return True
        except ValueError:
            return False


    def transform(self, doc_text, extra_features=None, idf=None):
        # first hashing vectorizer calculates integer token counts
        # (note that this uses a signed hash; negative indices are
        # are stored as a flipped (negated) value in the positive
        # index. This works fine so long as the model files use the
        # same rule (to balance out the negatives).

        sentences = [sent.text for sent in doc_text.sents]

        X_text = self.vectorizer.transform(sentences)

        X_rowsums = diags(X_text.sum(axis=1).A1, 0)
        if idf is not None:
            X_text = (X_text * idf) + X_text
            X_numeric = self.extract_numeric_features(doc_text, len(sentences))
            X_text.eliminate_zeros()

        if extra_features:
            X_extra_features = self.dict_vectorizer.transform(extra_features)
            # now combine feature sets.
            feature_matrix = sp.sparse.hstack((normalize(X_text), X_numeric, X_extra_features)).tocsr()
        else:
            #now combine feature sets.
            feature_matrix = sp.sparse.hstack((normalize(X_text), X_numeric)).tocsr()

        return feature_matrix



    def extract_numeric_features(self, doc_text, n, normalize_matrix=False):
        # number of numeric features (this is fixed
        # for now; may wish to revisit this)
        m = 12

        X_numeric = lil_matrix((n,m))#sp.sparse.csc_matrix((n,m))
        for sentence_index, sentence in enumerate(doc_text.sents):
            X_numeric[sentence_index, :] = self.extract_structural_features(sentence)
            # column-normalize
        X_numeric = X_numeric.tocsc()
        if normalize_matrix:
            X_numeric = normalize(X_numeric, axis=0)
        return X_numeric


    def extract_structural_features(self, sentence):

        fv = np.zeros(12)

        sent_text = sentence.text

        num_new_lines = sent_text.count("\n")
        if num_new_lines <= 1:
            fv[0] = 1
        elif num_new_lines < 20:
            fv[1] = 1
        elif num_new_lines < 40:
            fv[2] = 1
        else:
            fv[3] = 1

        line_lens = [len(line) for line in sent_text.split("\n") if not line.strip()==""]

        if line_lens:
            ##
            # maybe the *fraction* of lines less then... 10 chars?
            num_short_lines = float(len([len_ for len_ in line_lens if len_ <= 10]))
            frac_short_lines = float(num_short_lines)/float(len(line_lens))
        else:
            num_short_lines, frac_short_lines = 0, 0

        if frac_short_lines < .1:
            fv[4] = 1
        elif frac_short_lines <= .25:
            fv[5] = 1
        else:
            fv[6] = 1

        #fv[4] = 1 if frac_short_lines >= .25 else 0

        tokens = [w.text for w in sentence]
        num_numbers = sum([self.token_contains_number(t) for t in tokens])

        if num_numbers > 0:
            # i think you should replace with two indicators
            # 1 does it contain more than
            num_frac = num_numbers / float(len(tokens))
            # change to .1 and .3???
            #fv[2] = num_frac if num_frac > .2 else 0.0
            if num_frac < .2:
                fv[7] = 1
            elif num_frac < .4:
                fv[8] = 1
            else:
                # >= .4!
                fv[9] = 1

        if len(tokens):
            average_token_len = np.mean([len(t) for t in tokens])
            fv[10] = 1 if average_token_len < 5 else 0

        fv[11] = self.drugbank.contains_drug(sent_text)
        return fv



def main():
    # Sample code to make this run
    import unidecode, codecs, pprint
    # Read in example input to the text string
    with codecs.open('tests/example.txt', 'r', 'ISO-8859-1') as f:
        text = f.read()

    # make a PICO robot, use it to make predictions
    robot = PICORobot()
    annotations = robot.annotate(text)

    print("EXAMPLE OUTPUT:")
    print()
    pprint.pprint(annotations)


if __name__ == '__main__':
    main()
