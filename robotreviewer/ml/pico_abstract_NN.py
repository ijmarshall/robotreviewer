from collections import namedtuple

import numpy as np
import pandas as pd

from keras.models import Model, model_from_json

PicoData = namedtuple('PicoData', ['population', 'intervention', 'outcome'], verbose=True)

class PicoAbstractClassifier:
    def __init__(self, preprocessor, architecture_path=None, weights_path=None):
        '''
        Optionally allow a path to a (keras formatted) JSON model architecture
        specification and associated set of weights -- this allows easy loading
        and re-instantiation of trained models.
        '''
        self.preprocessor = preprocessor

        # check if we're loading in a pre-trained model
        if architecture_path is not None:
            assert(weights_path is not None)

            print("loading model architecture from file: %s" % architecture_path)
            with open(architecture_path) as model_arch:
                model_arch_str = model_arch.read()
                self.model = model_from_json(model_arch_str)

            self.model.load_weights(weights_path)

    def build_pico_model(self):

        #Need to attempt to load model from files.
        pass

    def predict_for_abstract(self, abstract_text):

        tokenized_abstract = tokenize_abstract(abstract_text)

        X = tokens_to_features(tokenized_abstract)

        pred = self.model.predict(X)

        #TODO Obviously should make a prediction at some point,
        #trying to get a skeleton setup first.

        return PicoData(None, None, None)
