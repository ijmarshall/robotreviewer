from collections import namedtuple
import os

import keras
import numpy as np
import pandas as pd
import nltk


from keras.models import Model, model_from_json, load_model
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss

import robotreviewer

PicoData = namedtuple('PicoData', ['population', 'intervention', 'outcome'], verbose=False)

class PicoAbstractClassifier:
    def __init__(self, architecture_path=None, weights_path=None):

        self.model = None
        self.word_to_index = None

    def build_pico_model(self):
        keras_model_path = os.path.join(robotreviewer.DATA_ROOT, "pico_abstract/full_model.hd5")
        word_to_index_path = os.path.join(robotreviewer.DATA_ROOT, "pico_abstract/word_to_index.npy")
        self.model = load_keras_model(keras_model_path)
        self.word_to_index = np.load(word_to_index_path).item()

    def predict_for_abstract(self, abstract_text):
        tokenized_abstract = tokenize_abstract(abstract_text)

        max_num_tokens = 30000

        max_abstract_len = 438
        # Extra 2 for the unk token and the blank token.
        num_regular_tokens = max_num_tokens
        num_tokens = num_regular_tokens + 2
        unk_token = num_tokens - 2
        blank_token = num_tokens - 1

        tokens_as_indices = [[self.word_to_index.get(token.lower(), unk_token) for token in tokenized_abstract]]
        data_for_predicting = keras.preprocessing.sequence.pad_sequences(
            tokens_as_indices, maxlen=max_abstract_len, value=blank_token, padding="post")


        pred = self.model.predict(data_for_predicting)
        
        pop_pred = spans_from_abstract(tokenized_abstract, np.argmax(pred[0][0], axis=1).tolist())
        intervention_pred = spans_from_abstract(tokenized_abstract, np.argmax(pred[1][0], axis=1).tolist())
        outcome_pred = spans_from_abstract(tokenized_abstract, np.argmax(pred[2][0], axis=1).tolist())

        print(pop_pred)
        print(intervention_pred)
        print(outcome_pred)


        return PicoData(pop_pred, intervention_pred, outcome_pred)


def create_custom_objects():
    return {"CRF": CRF, "crf_loss":crf_loss}

def load_keras_model(path):
    model = load_model(path, custom_objects=create_custom_objects())
    return model

def tokenize_abstract(abstract):
        """Take one abstract, break it up into sentences, then break up each sentence into words. Return array of tokenized words"""
        return [token for sentence in nltk.sent_tokenize(abstract) for token in nltk.word_tokenize(sentence)]

#  labels is a list of 1's and 0's
def spans_from_abstract(tokenized_abstract, labels):
    last_seen = 0
    current_span = []
    all_spans = []

    for token, label in zip(tokenized_abstract, labels):
        if label == 1:
            current_span.append(token)
        
        elif label == 0 and last_seen == 1:
            all_spans.append(current_span)
            current_span = []

        last_seen = label
    
    if current_span:
        all_spans.append(current_span)

    return list(set([" ".join(span) for span in all_spans]))

    