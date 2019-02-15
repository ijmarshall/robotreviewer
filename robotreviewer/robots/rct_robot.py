"""
the Randomized Control Trial (RCT) robot predicts whether a given
*abstract* (not full-text) describes an RCT.

    title =    '''Does usage of a parachute in contrast to free fall prevent major trauma?: a prospective randomised-controlled trial in rag dolls.'''
    abstract = '''PURPOSE: It is undisputed for more than 200 years that the use of a parachute prevents major trauma when falling from a great height. Nevertheless up to date no prospective randomised controlled trial has proven the superiority in preventing trauma when falling from a great height instead of a free fall. The aim of this prospective randomised controlled trial was to prove the effectiveness of a parachute when falling from great height. METHODS: In this prospective randomised-controlled trial a commercially acquirable rag doll was prepared for the purposes of the study design as in accordance to the Declaration of Helsinki, the participation of human beings in this trial was impossible. Twenty-five falls were performed with a parachute compatible to the height and weight of the doll. In the control group, another 25 falls were realised without a parachute. The main outcome measures were the rate of head injury; cervical, thoracic, lumbar, and pelvic fractures; and pneumothoraxes, hepatic, spleen, and bladder injuries in the control and parachute groups. An interdisciplinary team consisting of a specialised trauma surgeon, two neurosurgeons, and a coroner examined the rag doll for injuries. Additionally, whole-body computed tomography scans were performed to identify the injuries. RESULTS: All 50 falls-25 with the use of a parachute, 25 without a parachute-were successfully performed. Head injuries (right hemisphere p = 0.008, left hemisphere p = 0.004), cervical trauma (p < 0.001), thoracic trauma (p < 0.001), lumbar trauma (p < 0.001), pelvic trauma (p < 0.001), and hepatic, spleen, and bladder injures (p < 0.001) occurred more often in the control group. Only the pneumothoraxes showed no statistically significant difference between the control and parachute groups. CONCLUSIONS: A parachute is an effective tool to prevent major trauma when falling from a great height.'''
    ptyp = ["Randomized Controlled Trial", "Intervention Study", "Journal Article"]

    rct_robot = RCTRobot()
    annotations = rct_robot.annotate(title, abstract, ptyp)

This model was trained on the Cochrane crowd dataset, and validated on the Clinical Hedges dataset
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
from robotreviewer.parsers import ris

from collections import Counter
from scipy.sparse import lil_matrix, hstack

import numpy as np
import re
import glob
from sklearn.feature_extraction.text import VectorizerMixin
from sklearn.base import ClassifierMixin



class KerasVectorizer(VectorizerMixin):
    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 analyzer='word', embedding_inits=None, vocab_map_file=None):
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.ngram_range = (1, 1)
        self.embedding_inits = embedding_inits # optional gensim word2vec model
        self.embedding_dim = embedding_inits.syn0.shape[1] if embedding_inits else None
        with open(vocab_map_file, 'rb') as f:
            self.vocab_map = pickle.load(f)

    def transform(self, raw_documents, maxlen=400):
        """
        returns lists of integers
        """
        analyzer = self.build_analyzer()
        int_lists = [[1]+[self.vocab_map.get(w, 2) for w in analyzer(t)] for t in raw_documents]
        # 0 = pad, 1 = start, 2 = OOV
        return sequence.pad_sequences(int_lists, maxlen=maxlen)





class RCTRobot:

    def __init__(self):
        from keras.preprocessing import sequence
        from keras.models import load_model
        from keras.models import Sequential
        from keras.preprocessing import sequence
        from keras.layers import Dense, Dropout, Activation, Lambda, Input, merge, Flatten
        from keras.layers import Embedding
        from keras.layers import Convolution1D, MaxPooling1D
        from keras import backend as K
        from keras.models import Model
        from keras.regularizers import l2
        global sequence, load_model, Sequential, Dense, Dropout, Activation, Lambda, Input, merge, Flatten
        global Embedding, Convolution1D, MaxPooling1D, K, Model, l2
        self.svm_clf = MiniClassifier(os.path.join(robotreviewer.DATA_ROOT, 'rct/rct_svm_weights.npz'))
        cnn_weight_files = glob.glob(os.path.join(robotreviewer.DATA_ROOT, 'rct/*.h5'))
        self.cnn_clfs = [load_model(cnn_weight_file) for cnn_weight_file in cnn_weight_files]
        self.svm_vectorizer = HashingVectorizer(binary=False, ngram_range=(1, 1), stop_words='english')
        self.cnn_vectorizer = KerasVectorizer(vocab_map_file=os.path.join(robotreviewer.DATA_ROOT, 'rct/cnn_vocab_map.pck'), stop_words='english')
        with open(os.path.join(robotreviewer.DATA_ROOT, 'rct/rct_model_calibration.json'), 'r') as f:
            self.constants = json.load(f)

    def _process_ptyp(self, data_row, strict=True):
        """
        Takes in a data row which might include rct_ptyp
        or ptyp fields.
        If strict=True, then raises exception when passed any
        contradictory data
        Returns: 1 = ptyp is RCT
                 0 = ptyp is NOT RCT
                 -1 = no ptyp information present
        """
        if data_row['use_ptyp'] == False:
            return -1
        elif data_row['use_ptyp'] == True:
            return 1 if any((tag in data_row['ptyp'] for tag in ["Randomized Controlled Trial", "D016449"])) else 0
        else:
            raise Exception("unexpcted value for 'use_ptyp'")


    def annotate(self, data):

        # use the best performing models from the validation paper (in draft...)
        filter_class = "svm_cnn"
        threshold_class = "balanced"
        auto_use_ptyp=True

        if data.get("abstract") is not None and data.get("title") is not None:
            ti = data["title"]
            ab = data["abstract"]
        elif data.get("parsed_text") is not None:
            # then just use the start of the document
            TI_LEN = 30
            AB_LEN = 500
            # best guesses based on sample of RCT abstracts + aiming for 95% centile
            ti = data['parsed_text'][:TI_LEN].text
            ab = data['parsed_text'][:AB_LEN].text
        else:
            # else can't proceed
            return data

        # ignore PubMed data for now
        # if "pubmed" in data.data:
        #     ptyp = 1.0
        # else:
        #     ptyp = 0.0

        preds = self.predict({"title": ti, "abstract": ab}, auto_use_ptyp=False)[0]


        structured_data = {"is_rct": preds['is_rct'],
                           "decision_score": preds['threshold_value'],
                           "model_class": preds['model'],
                           "filter_type": preds['threshold_type']}

        data.ml["rct"] = structured_data
        return data

    def predict(self, X, filter_class="svm", filter_type="sensitive", auto_use_ptyp=True):


        if isinstance(X, dict):
            X = [X]

        if auto_use_ptyp:
            pt_mask = np.array([self._process_ptyp(r) for r in X])
        else:
            # don't add for any of them
            pt_mask = np.array([-1 for r in X])

        preds_l = {}
        # calculate ptyp for all
        #ptyp = np.copy(pt_mask)
        # ptyp = np.array([(article.get('rct_ptyp')==True)*1. for article in X])
        ptyp_scale = (pt_mask - self.constants['scales']['ptyp']['mean']) / self.constants['scales']['ptyp']['std']
        # but set to 0 if not using
        ptyp_scale[pt_mask==-1] = 0
        preds_l['ptyp'] = ptyp_scale

        # thresholds vary per article
        thresholds = []
        for r in pt_mask:
            if r != -1:
                thresholds.append(self.constants['thresholds']["{}_ptyp".format(filter_class)][filter_type])
            else:
                thresholds.append(self.constants['thresholds'][filter_class][filter_type])

        X_ti_str = [article.get('title', '') for article in X]
        X_ab_str = ['{}\n\n{}'.format(article.get('title', ''), article.get('abstract', '')) for article in X]

        if "svm" in filter_class:
            X_ti = lil_matrix(self.svm_vectorizer.transform(X_ti_str))
            X_ab = lil_matrix(self.svm_vectorizer.transform(X_ab_str))
            svm_preds = self.svm_clf.decision_function(hstack([X_ab, X_ti]))
            svm_scale =  (svm_preds - self.constants['scales']['svm']['mean']) / self.constants['scales']['svm']['std']
            preds_l['svm'] = svm_scale
            preds_l['svm_ptyp'] = preds_l['svm'] + preds_l['ptyp']

        if "cnn" in filter_class:
            X_cnn = self.cnn_vectorizer.transform(X_ab_str)
            cnn_preds = []
            for i, clf in enumerate(self.cnn_clfs):
                cnn_preds.append(clf.predict(X_cnn).T[0])

            cnn_preds = np.vstack(cnn_preds)
            cnn_scale =  (cnn_preds - self.constants['scales']['cnn']['mean']) / self.constants['scales']['cnn']['std']
            preds_l['cnn'] = np.mean(cnn_scale, axis=0)

            preds_l['cnn_ptyp'] = preds_l['cnn'] + preds_l['ptyp']

        if filter_class == "svm_cnn":
            weights = [self.constants['scales']['svm']['weight']] + ([self.constants['scales']['cnn']['weight']] * len(self.cnn_clfs))
            preds_l['svm_cnn'] = np.average(np.vstack([svm_scale, cnn_scale]), axis=0, weights=weights)


            preds_l['svm_cnn_ptyp'] = preds_l['svm_cnn'] + preds_l['ptyp']


        preds_d =[dict(zip(preds_l,i)) for i in zip(*preds_l.values())]

        out = []
        for pred, threshold, used_ptyp in zip(preds_d, thresholds, pt_mask):
            row = {}
            if used_ptyp != -1:
                row['model'] = "{}_ptyp".format(filter_class)
            else:
                row['model'] = filter_class
            row['score'] = float(pred[row['model']])
            row['threshold_type'] = filter_type
            row['threshold_value'] = float(threshold)
            row['is_rct'] = bool(row['score'] >= threshold)
            row['ptyp_rct'] = int(used_ptyp)
            row['preds'] = {k: float(v) for k, v in pred.items()}
            out.append(row)
        return out



    @staticmethod
    def get_marginalia(data):
        """
        Get marginalia formatted for Spa from structured data
        """
        marginalia = [{"type": "Trial Design",
                      "title": "Is an RCT?",
                      "annotations": [],
                      "description":  "{0} (Decision score={1:0.2f} using {} model)".format(data["rct"]["is_rct"], data["rct"]["decision_score"], data["rct"]["model_class"])}]
        return marginalia


    def predict_ris(self, ris_data, filter_class="svm", filter_type='sensitive', auto_use_ptyp=False):


        simplified = [ris.simplify(article) for article in ris_data]
        preds = self.predict(simplified, filter_class=filter_class, filter_type=filter_type, auto_use_ptyp=auto_use_ptyp)
        return preds


    def filter_articles(self, ris_string, filter_class="svm", filter_type='sensitive', auto_use_ptyp=True, remove_non_rcts=True):

        print('Parsing RIS data')
        ris_data = ris.loads(ris_string)
        import json
        with open("debug.json", 'w') as f:
            json.dumps(ris_data)
        preds = self.predict_ris(ris_data, filter_class=filter_class, filter_type=filter_type, auto_use_ptyp=auto_use_ptyp)
        out = []

        pred_key_map = {"score": "ZS", "model": "ZM", "threshold_type": "ZT", "threshold_value": "ZC", "is_rct": "ZR", "ptyp_rct": "ZP"}

        for ris_row, pred_row in zip(ris_data, preds):
            if remove_non_rcts==False or pred_row['is_rct']:
                ris_row.update({pred_key_map[k]: v for k, v in pred_row.items()})

                out.append(ris_row)
        return ris.dumps(out)


def test_calibration():
    print("Testing RobotSearch...")
    target_classes = ["svm", "cnn", "svm_cnn"]
    target_modes = ["balanced", "precise", "sensitive"]

    rct_bot = RCTRobot()

    print("Loading test PubMed file")
    with open(os.path.join(robotreviewer.DATA_ROOT, 'rct/pubmed_test.txt'), 'r') as f:
        ris_string = f.read()


    print('Parsing RIS data')
    ris_data = ris.loads(ris_string)

    print("Loading expected results (from validation paper)")
    with open(os.path.join(robotreviewer.DATA_ROOT, 'rct/pubmed_expected.json'), 'r') as f:
        expected_results = json.load(f)



    for target_class in target_classes:
        for target_mode in target_modes:
            for use_ptyp in [True, False]:

                expected_model_class = "{}_ptyp".format(target_class) if use_ptyp else target_class

                print("Testing {} model; use_ptyp={}; mode={}".format(target_class, use_ptyp, target_mode))
                data = rct_bot.predict_ris(ris_data, filter_class=target_class, filter_type=target_mode, auto_use_ptyp=use_ptyp)

                exp_pmids = [str(r['PMID'][0]) for r in ris_data]
                obs_pmids = [str(r['pmid']) for r in expected_results[expected_model_class][target_mode]]


                print("Number matching PMIDS: {}".format(sum([i==j for i, j in zip(exp_pmids, obs_pmids)])))


                obs_score = np.array([r['score'] for r in data])
                obs_clf = np.array([r['is_rct'] for r in data])



                exp_score = np.array([float(r['score']) for r in expected_results[expected_model_class][target_mode]])
                exp_clf = np.array([r['is_rct'] for r in expected_results[expected_model_class][target_mode]])

                print("Totals assessed: {} obs, {} exp".format(len(obs_score), len(exp_score)))
                match_clf = np.sum(np.equal(obs_clf, exp_clf))



                disag = np.where((np.equal(obs_clf, exp_clf)==False))[0]
                hedges_y = np.array([r['hedges_is_rct']=='1' for r in expected_results[expected_model_class][target_mode]])


                exp_sens = np.sum(exp_clf[hedges_y])/np.sum(hedges_y)
                exp_spec = np.sum(np.invert(exp_clf)[np.invert(hedges_y)])/np.sum(np.invert(hedges_y))


                obs_sens = np.sum(obs_clf[hedges_y])/np.sum(hedges_y)
                obs_spec = np.sum(np.invert(obs_clf)[np.invert(hedges_y)])/np.sum(np.invert(hedges_y))

                print("Expected: sens {} spec {}".format(exp_sens, exp_spec))

                print("Observed: sens {} spec {}".format(obs_sens, obs_spec))
