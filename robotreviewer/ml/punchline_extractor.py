import random
import sys
from os.path import join, dirname, abspath
sys.path.insert(0, abspath(join(dirname(abspath(__file__)), '..', '..')))

import numpy as np 

import keras
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Model, model_from_json
from keras.callbacks import ModelCheckpoint 

from bert_serving.client import BertClient

BERT_HIDDEN_DIM = 768
BERT_IP = "bert"
BERT_PORT = 5555
BERT_PORT_OUT = 5556


class PunchlineExtractor:

    def __init__(self, architecture_path=None, weights_path=None):
        self.bc = None
        try: 
            self.bc = BertClient(ip=BERT_IP, port=BERT_PORT, port_out=BERT_PORT_OUT)
        except:
            raise Exception("PunchlineExtractor: Cannot instantiate BertClient. Is it running???")

        # check if we're loading in a pre-trained model
        if architecture_path is not None:
            assert(weights_path is not None)
            
            with open(architecture_path) as model_arch:
                model_arch_str = model_arch.read()
                self.model = model_from_json(model_arch_str)

            self.model.load_weights(weights_path)
        else:
            self.build_model()


    def build_model(self):
        BERT_features = Input(shape=(BERT_HIDDEN_DIM, ))
        x = Dense(256, name="hidden1", activation='relu')(BERT_features)
        x = Dense(128, name="hidden2", activation='relu')(x)
        y_hat = Dense(1, name="prediction", activation="sigmoid")(x)

        self.model = Model([BERT_features], output=[y_hat])
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    def score_sentence(self, sent):
        x = self.bc.encode([sent])
        y_hat = self.model.predict(x)
        return y_hat

    def score_sentences(self, sentences):
        x = self.bc.encode(sentences)
        y_hat = self.model.predict(x)
        return y_hat


class SimpleInferenceNet:

    def __init__(self, architecture_path=None, weights_path=None):
        self.bc = None
        try: 
            self.bc = BertClient(ip=BERT_IP, port=BERT_PORT, port_out=BERT_PORT_OUT)
        except:
            raise Exception("PunchlineExtractor: Cannot instantiate BertClient. Is it running???")

        # check if we're loading in a pre-trained model
        if architecture_path is not None:
            assert(weights_path is not None)
            
            with open(architecture_path) as model_arch:
                model_arch_str = model_arch.read()
                self.model = model_from_json(model_arch_str)

            self.model.load_weights(weights_path)
        else:
            self.build_model()

    def build_model(self):
        BERT_features = Input(shape=(BERT_HIDDEN_DIM, ))
        x = Dense(256, name="hidden1", activation='relu')(BERT_features)
        x = Dropout(0.2)(x)
        x = Dense(128, name="hidden2", activation='relu')(x)
        y_hat = Dense(3, name="prediction", activation="softmax")(x)

        self.model = Model([BERT_features], output=[y_hat])
        self.model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy"])

    def infer_result(self, sent):
       x = self.bc.encode(sent)
       y_hat = self.model.predict(x)
       return y_hat


def make_Xy(Xy_dicts, BERT_client, neg_samples=5, min_len=3):
    X, y = [], []
    for Xy_dict in Xy_dicts:
        y_rationales = [y_i[1] for y_i in Xy_dict['y'] if len(y_i[1].split(" ")) >= min_len] 
        if len(y_rationales) > 0:
            #[Xy_dict['y'][j][1] for j in Xy_dict['y'] 
            y_rationale = random.sample(y_rationales, 1)
            pos_X = BERT_client.encode(y_rationale)
            X.extend(pos_X)
            y.append(1)
            TEXT_IDX = 2 # actual index of the sentence texts
            neg_sents = filtered_sents = [s[TEXT_IDX] for s in Xy_dict['all_article_sentences'] if len(s[TEXT_IDX].split(" ")) >= min_len]
            if neg_samples < len(filtered_sents):
                neg_sents = random.sample(filtered_sents, neg_samples)
            neg_X = BERT_client.encode(neg_sents)
            X.extend(neg_X)
            y.extend([0]*len(neg_sents))

    X = np.array(X)
    y = np.array(y)
    return X, y


def convert_to_sparse(lbl):
    # sig decrease, no diff, sig increase
    y_sp = np.zeros(3)
    y_sp[lbl+1] = 1
    return y_sp

def make_Xy_inference(Xy_dicts, BERT_client, min_len=3):
    X, y = [], []
    for Xy_dict in Xy_dicts:
        y_lbls_and_rationales = [(y_i[0], y_i[1]) for y_i in Xy_dict['y'] if len(y_i[1].split(" ")) >= min_len]
        if len(y_lbls_and_rationales) > 0:
            y_i, rationale = random.sample(y_lbls_and_rationales, 1)[0]
            x = BERT_client.encode([rationale])
            X.append(x)
            y.append(convert_to_sparse(y_i))


    X = np.array(X).squeeze()
    y = np.array(y)
    return X, y

def train():
    # train the model -- this assumes access to evidence_inference:
    # https://github.com/jayded/evidence-inference/tree/master/evidence_inference
    # which is not needed in general to load the trained model.
    #
    # if inference_true flag is on, then a model will also be fit that predicts the
    # outcome (sig. decrease, no diff, sig. increase) given punchline snippets.
    from evidence_inference.preprocess.preprocessor import get_Xy, train_document_ids, test_document_ids, validation_document_ids, get_train_Xy

    extractor_model = PunchlineExtractor()

    tr_ids, val_ids, te_ids = train_document_ids(), validation_document_ids(), test_document_ids()
    tr_ids = list(train_document_ids())
    train_Xy, inference_vectorizer = get_train_Xy(tr_ids, sections_of_interest=None, vocabulary_file=None, include_sentence_span_splits=False, include_raw_texts=True)
    # Create vectors and targets for extraction task
    X_k, y_k = make_Xy(train_Xy, extractor_model.bc)
    print("train data loaded!") 
 
    val_Xy = get_Xy(val_ids, inference_vectorizer,  include_raw_texts=True)    
    X_kv, y_kv = make_Xy(val_Xy, extractor_model.bc, neg_samples=1)
    print("val data loaded!") 

    # Fit the model!
    filepath="punchline.weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
   
    with open("punchline_model.json", "w") as outf:
       outf.write(extractor_model.model.to_json())
    
    print("fitting punchline extractor!")
    extractor_model.model.fit(X_k, y_k, validation_data=(X_kv, y_kv), callbacks=callbacks_list, epochs=50)


def train_simple_inference_net(n_epochs=30):
    inf_net = SimpleInferenceNet()
    tr_ids, val_ids, te_ids = train_document_ids(), validation_document_ids(), test_document_ids()
    tr_ids = list(train_document_ids())
    train_Xy, inference_vectorizer = get_train_Xy(tr_ids, sections_of_interest=None, vocabulary_file=None, include_sentence_span_splits=False, include_raw_texts=True)

    X_k, y_k = make_Xy_inference(train_Xy, inf_net.bc)
    print("train data for inference task loaded!")

    val_Xy = get_Xy(val_ids, inference_vectorizer,  include_raw_texts=True)
    X_kv, y_kv = make_Xy_inference(val_Xy, inf_net.bc)
    print("val data loaded!")

    filepath="inference.weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    with open("inference_model.json", "w") as outf:
        outf.write(inf_net.model.to_json())

    print("fitting inference model!")
    inf_net.model.fit(X_k, y_k, validation_data=(X_kv, y_kv), callbacks=callbacks_list, epochs=n_epochs)



   
if __name__ == "__main__":
    print("calling train...")
    train()
