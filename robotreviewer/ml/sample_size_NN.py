import operator 
import pickle

import numpy as np 
import pandas as pd 
import spacy 

import gensim 
from gensim.models import Word2Vec

import keras 
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers import merge
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, model_from_json
from keras.preprocessing.text import text_to_word_sequence, Tokenizer

import index_numbers

class MLPSampleSizeClassifier:

    def __init__(self, preprocessor, architecture_path=None, weights_path=None):
        '''
        Optionally allow a path to a (kera's formatted) JSON model architecture
        specification and associated set of weights -- this allows easy loading
        and re-instantiation of trained models.
        '''
        self.preprocessor = preprocessor
        # threshold governing whether to abstain from predicting
        # this as a sample size altogether (for highest scoring 
        # integer). As always, this was definitely set in a totally
        # scientifically sound way ;).
        self.nlp = spacy.load('en') # to avoid instantiating multiple times.
        self.number_tagger = index_numbers.NumberTagger()

        # check if we're loading in a pre-trained model
        if architecture_path is not None: 
            assert(weights_path is not None)

            print("loading model architecture from file: %s" % architecture_path)
            with open(architecture_path) as model_arch:
                model_arch_str = model_arch.read()
                self.model = model_from_json(model_arch_str)
            
            self.model.load_weights(weights_path)


    def featurize_for_input(self, X):
        left_token_inputs, target_token_inputs, right_token_inputs, other_inputs = [], [], [], []

        # helper func for looking up word indices
        def get_w_index(w):
            unk_idx = self.preprocessor.tokenizer.word_index[self.preprocessor.unk_symbol]
            try: 
                word_idx = self.preprocessor.tokenizer.word_index[w]
                if word_idx <= self.preprocessor.max_features:
                    return word_idx 
                else: 
                    return unk_idx
            except: 
                return unk_idx


        for x in X:
            l_word_idx = get_w_index(x["left word"])
            left_token_inputs.append(np.array([l_word_idx]))

            target_token_inputs.append(np.array(x["target"]))

            r_word_idx = get_w_index(x["right word"])
            right_token_inputs.append(np.array(r_word_idx))

            other_inputs.append(np.array(x["other features"]))

        
        X_inputs_dict = {"left token input":np.vstack(left_token_inputs), 
                        "target token input":np.vstack(target_token_inputs),
                        "right token input":np.vstack(right_token_inputs),
                        "other feature inputs":np.vstack(other_inputs)}

        return X_inputs_dict
    

    def build_MLP_model(self):

        left_token_input = Input(name='left token input', shape=(1,))
        left_token_embedding = Embedding(output_dim=self.preprocessor.embedding_dims, input_dim=self.preprocessor.max_features, 
                                        input_length=1)(left_token_input)
        left_token_embedding = Flatten(name="left token embedding")(left_token_embedding)
                      
        target_token_input = Input(name='target token input', shape=(1,))

        right_token_input = Input(name='right token input', shape=(1,))
        right_token_embedding = Embedding(output_dim=self.preprocessor.embedding_dims, input_dim=self.preprocessor.max_features, 
                                          input_length=1)(right_token_input)
        right_token_embedding = Flatten(name="right token embedding")(right_token_embedding)

        one_hot_features_input = Input(name='other feature inputs', shape=(1,))

        #x = Merge([left_token_embedding, target_token_embedding, right_token_embedding,  one_hot_features_input], mode='concat')
        x = merge([left_token_embedding, target_token_input, right_token_embedding,  one_hot_features_input],  
                        mode='concat', concat_axis=1)
        x = Dense(128, name="hidden 1", activation='relu')(x)
        x = Dense(64, name="hidden 2", activation='relu')(x) 

        output = Dense(1, name="prediction", activation='sigmoid')(x)

        self.model = Model([left_token_input, target_token_input, right_token_input, one_hot_features_input], 
                           output=[output])

        self.model.compile(optimizer="adam", loss="binary_crossentropy")

    def predict_for_abstract(self, abstract_text):
        ''' 
        returns either the extracted sample size, or None if this cannot
        be located. 
        '''
        abstract_text_w_numbers = self.number_tagger.swap(abstract_text)
        abstract_tokens = tokenize_abstract(abstract_text_w_numbers, self.nlp)
        abstract_features, numeric_token_indices = abstract2features(abstract_tokens)

        # no numbers in the abstract, then!
        if len(abstract_features) == 0: 
            return None 

        X = self.featurize_for_input(abstract_features)
        preds = self.model.predict(X)
        most_likely_idx = np.argmax(preds)
        
        return (preds[most_likely_idx][0], abstract_tokens[numeric_token_indices[most_likely_idx]])

    


def load_trained_w2v_model(path):
    m = Word2Vec.load_word2vec_format(path, binary=True)
    return m

class Preprocessor:
    def __init__(self, max_features, wvs, all_texts, unk=True, unk_symbol="unkunk"):
        '''
        max_features: the upper bound to be placed on the vocabulary size.
        embedding_dims: size of the token embeddings; over-ridden if pre-trained
                          vectors is provided (if wvs is not None).
        wvs: set of word vectors to be used for initialization
        '''
        self.unk = unk 
        self.unk_symbol = unk_symbol
        self.max_features = max_features  
        self.tokenizer = Tokenizer(nb_words=self.max_features) 

        self.embedding_dims = wvs.vector_size
        self.word_embeddings = wvs

        self.raw_texts = all_texts
        self.unked_texts = []
        self.fit_tokenizer()
       
        if self.unk:
            # rewrite the 'raw texts' with unked versions, where tokens not in the
            # top max_features are unked.
            sorted_tokens = sorted(self.tokenizer.word_index, key=self.tokenizer.word_index.get)
            self.known_tokens = sorted_tokens[:self.max_features]
            self.tokens_to_unk = sorted_tokens[self.max_features:]

            for idx, text in enumerate(self.raw_texts):
                cur_text = text_to_word_sequence(text, split=self.tokenizer.split)
                t_or_unk = lambda t : t if t in self.known_tokens else self.unk_symbol
                unked_text = [t_or_unk(t) for t in cur_text]
                unked_text = self.tokenizer.split.join(unked_text)

                self.unked_texts.append(unked_text)

            self.raw_texts = self.unked_texts
            self.fit_tokenizer()

        self.init_word_vectors()


    def fit_tokenizer(self):
        ''' Fits tokenizer to all raw texts; remembers indices->words mappings. '''
        self.tokenizer.fit_on_texts(self.raw_texts)
        self.word_indices_to_words = {}
        for token, idx in self.tokenizer.word_index.items():
            self.word_indices_to_words[idx] = token

    def init_word_vectors(self):
        ''' 
        Initialize word vectors.
        '''
        self.init_vectors = []
        unknown_words_to_vecs = {}
        for t, token_idx in self.tokenizer.word_index.items():
            if token_idx <= self.max_features:
                try:
                    self.init_vectors.append(self.word_embeddings[t])
                except:
                    if t not in unknown_words_to_vecs:
                        # randomly initialize
                        unknown_words_to_vecs[t] = np.random.random(
                                                self.embedding_dims)*-2 + 1

                    self.init_vectors.append(unknown_words_to_vecs[t])



        # note that we make this a singleton list because that's
        # what Keras consumes.
        self.init_vectors = [np.vstack(self.init_vectors)]

def y_to_bin(y):
    y_bin = np.zeros(len(y))
    for idx, y_i in enumerate(y):
        if y_i == "N":
            y_bin[idx] = 1.0
    return y_bin

def _is_an_int(s):
    try:
        int(s) 
        return True
    except:
        return False

def get_X_y(df):
    nlp = spacy.load('en') # for POS tagging

    X, y = [], []
    for instance in df.iterrows():
        instance = instance[1]
  
        abstract_tokens = tokenize_abstract(instance["ab_numbers"], nlp)
        
        nums_to_labels = {instance["enrolled_totals"]:"N", instance["enrolled_P1"]:"n1", instance["enrolled_P2"]:"n2"}
        cur_y = annotate(abstract_tokens, nums_to_labels)
        cur_x, numeric_token_indices = abstract2features(abstract_tokens)

        X.extend(cur_x)
        y.extend([cur_y[idx] for idx in numeric_token_indices])
        
    return X, y_to_bin(y)

def annotate(tokenized_abstract, nums_to_labels):
    # nums_to_labels : dictionary mapping numbers to labels
    y = []
    for t in tokenized_abstract:
        try: 
            t_num = int(t)
            if t_num in nums_to_labels.keys():
                y.append(nums_to_labels[t_num])
            else:
                y.append("O")
        except:
            y.append("O")
    return y 


def tokenize_abstract(abstract, nlp=None):
    if nlp is None:
        nlp = spacy.load('en')

    tokens = []
    ab = nlp(abstract)
    for word in ab:
        tokens.append(word.text)
        
    return tokens

def abstract2features(abstract_tokens):
    # we use all the numbers to induce global features
    all_nums_in_abstract = []
    for t in abstract_tokens:
        try:
            num = int(t)
            all_nums_in_abstract.append(num)
        except:
            pass

    # note that we keep track of all candidates/numbers
    # and pass this back.
    x, numeric_token_indices = [], []
    for word_idx in range(len(abstract_tokens)):
        if (_is_an_int(abstract_tokens[word_idx])):   
            numeric_token_indices.append(word_idx)         
            features = word2features(abstract_tokens, word_idx, all_nums_in_abstract)
            x.append(features)

    return x, numeric_token_indices


def word2features(abstract_tokens, i, all_nums_in_abstract):
    l_word, r_word = "", ""
    t_word = abstract_tokens[i]

    if i > 0:
        l_word = abstract_tokens[i-1].lower()
    else:
        l_word = "BoS"

    if i < len(abstract_tokens)-1:
        r_word = abstract_tokens[i+1].lower()
    else: 
        r_word = "LoS"

    target_num = int(t_word)
    # need to add a feature for being largest in doc??
    biggest_num_in_abstract = 0.0
    if target_num >= max(all_nums_in_abstract):
        biggest_num_in_abstract = 1.0
    return {"left word":l_word, "target": target_num, "right word":r_word, "other features":[biggest_num_in_abstract]}

def load_data(csv_path="ctgov_with_tags.csv"):
    return pd.read_csv(csv_path)








