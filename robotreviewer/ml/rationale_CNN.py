'''
@authors Byron Wallace, Edward Banner, Ye Zhang, Iain Marshall

A Keras implementation of our "rationale augmented CNN" (https://arxiv.org/abs/1605.04469); see
reference below.

Please note that the model was originally implemented in Theano; results reported in the paper 
are from said implementation. This version is a work in progress. 

Credit for initial pass of basic CNN implementation to: Cheng Guo (https://gist.github.com/entron).
For a more basic implementation of CNN for text classification, see: 
https://github.com/bwallace/CNN-for-text-classification

References
--
Ye Zhang, Iain J. Marshall and Byron C. Wallace. "Rationale-Augmented Convolutional Neural Networks for Text Classification". http://arxiv.org/abs/1605.04469
Yoon Kim. "Convolutional Neural Networks for Sentence Classification". EMNLP 2016.
Ye Zhang and Byron Wallace. "A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification". http://arxiv.org/abs/1510.03820.
& c.f. http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
'''

from __future__ import print_function
import pdb
import sys 
try:
    reload(sys)
    sys.setdefaultencoding('utf8')
except:
    # almost certainly means Python 3x
    pass 

import random


import numpy as np

from keras.optimizers import SGD, RMSprop
from keras import backend as K 
K.set_image_dim_ordering("th")
K.set_image_data_format("channels_first")

from keras.models import Model, Sequential, model_from_json #load_model
from keras.preprocessing import sequence
from keras.engine.topology import Layer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Dense, merge
from keras.layers.merge import concatenate
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute, Lambda
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, Convolution2D, Conv2D, MaxPooling1D, MaxPooling2D
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.constraints import maxnorm
from keras.regularizers import l2


from celery.contrib import rdb

class RationaleCNN:

    def __init__(self, preprocessor, filters=None, n_filters=32, 
                        sent_dropout=0.5, doc_dropout=0.5, 
                        end_to_end_train=False, f_beta=2,
                        document_model_architecture_path=None,
                        document_model_weights_path=None):
        '''
        parameters
        ---
        preprocessor: an instance of the Preprocessor class, defined below
        '''
        self.preprocessor = preprocessor

        if filters is None:
            self.ngram_filters = [3, 4, 5]
        else:
            self.ngram_filters = filters 

      
        self.n_filters = n_filters 
        self.sent_dropout = sent_dropout
        self.doc_dropout  = doc_dropout
        self.sentence_model_trained = False 
        self.end_to_end_train = end_to_end_train
        self.sentence_prob_model = None 
        self.f_beta = f_beta

        if document_model_architecture_path is not None: 
            assert(document_model_weights_path is not None)

            print("loading model architecture from file: %s" % document_model_architecture_path)

            with open(document_model_architecture_path) as doc_arch:
                doc_arch_str = doc_arch.read()
                self.doc_model = model_from_json(doc_arch_str)
            
            self.doc_model.load_weights(document_model_weights_path)

            self.set_final_sentence_model() # setup sentence model, too
            print("ok!")


    @staticmethod
    def metric_func_maker(metric_name="f", beta=1):
        
        return_recall=False
        return_precision=False
        
        func_name = metric_name
        if metric_name == "recall": 
            return_recall = True 
        elif metric_name == "precision":
            return_precision = True 
        else: 
            func_name = "f_%s" % beta
            
        def f_beta_score(y, y_pred):
            ''' for convienence '''
            y_pred_binary = K.round(y_pred)
            num_true = K.sum(y)
            num_pred = K.sum(y_pred_binary)
            tp = K.sum(y * y_pred_binary)

            recall = K.switch(num_true>0, tp / num_true, 0.0)
            if return_recall:
                return recall

            precision = K.switch(num_pred>0, tp / num_pred, 0.0)
            if return_precision:
                return precision 

            precision_recall_sum = recall + (beta*precision)

            return K.switch(precision_recall_sum>0, 
                             (beta+1)*((precision*recall)/(precision_recall_sum)), 0.0)


        f_beta_score.__name__ = func_name
        return f_beta_score

    @staticmethod
    def get_weighted_sum_func(X, weights):
        # @TODO.. add sentence preds!
        def weighted_sum(X):
            return K.sum(np.multiply(X, weights), axis=-1)
        
        #return K.sum(X, axis=0) 
        return weighted_sum

    @staticmethod
    def weighted_sum_output_shape(input_shape):
        # expects something like (None, max_doc_len, num_features) 
        # returns (1 x num_features)
        shape = list(input_shape)
        return tuple((1, shape[-1]))

    @staticmethod
    def balanced_sample(X, y, sentences=None, binary=False, k=1, n_rows=None):
        if binary:
            _, neg_indices = np.where([y <= 0]) 
            _, pos_indices = np.where([y > 0])
            sampled_neg_indices = np.random.choice(neg_indices, pos_indices.shape[0], replace=False)
            train_indices = np.concatenate([pos_indices, sampled_neg_indices])
        else:        
            _, pos_rationale_indices = np.where([y[:,0] > 0]) 
            _, neg_rationale_indices = np.where([y[:,1] > 0]) 
            _, non_rationale_indices = np.where([y[:,2] > 0]) 


            if n_rows is not None: 
                # then we will return a matrix comprising n_rows rows, 
                # repeating positive examples but drawing diverse negative
                # instances
                num_rationale_indices = int(n_rows / 2.0)
                if pos_rationale_indices.shape[0] > 0:
                    rationale_indices = np.random.choice(pos_rationale_indices, num_rationale_indices, replace=True)
                else: 
                    rationale_indices = np.random.choice(neg_rationale_indices, num_rationale_indices, replace=True)

                # sample the rest as `negative' (neutral) instances
                num_non_rationales = n_rows - num_rationale_indices
                sampled_non_rationale_indices = np.random.choice(non_rationale_indices, num_non_rationales, replace=True)
                train_indices = np.concatenate([rationale_indices, sampled_non_rationale_indices])
                
            else:

                # sample a number of non-rationales equal to the total
                # number of pos/neg rationales * k
                m = k*(pos_rationale_indices.shape[0] + neg_rationale_indices.shape[0])
                                                # np.array(random.sample(non_rationale_indices, m)) 

                sampled_non_rationale_indices = non_rationale_indices
                if m < non_rationale_indices.shape[0]:
                    sampled_non_rationale_indices = np.random.choice(non_rationale_indices, m, replace=True)

                train_indices = np.concatenate([pos_rationale_indices, neg_rationale_indices, 
                                                    sampled_non_rationale_indices])
            


        np.random.shuffle(train_indices) # why not
        if sentences is not None: 
            return X[train_indices,:], y[train_indices], [sentences[idx] for idx in train_indices]
        return X[train_indices,:], y[train_indices]


    def build_simple_doc_model(self):
        # maintains sentence structure, but does not impose weights.
        tokens_input = Input(name='input', 
                            shape=(self.preprocessor.max_doc_len, self.preprocessor.max_sent_len), 
                            dtype='int32')

        tokens_reshaped = Reshape([self.preprocessor.max_doc_len*self.preprocessor.max_sent_len])(tokens_input)

    
        x = Embedding(self.preprocessor.max_features+1, self.preprocessor.embedding_dims, 
                        weights=self.preprocessor.init_vectors,
                        name="embedding")(tokens_reshaped)

        x = Reshape((1, self.preprocessor.max_doc_len, 
                     self.preprocessor.max_sent_len*self.preprocessor.embedding_dims), 
                     name="reshape")(x)

        convolutions = []

        for n_gram in self.ngram_filters:
            cur_conv = Convolution2D(self.n_filters, 1, 
                                     n_gram*self.preprocessor.embedding_dims, 
                                     subsample=(1, self.preprocessor.embedding_dims),
                                     activation="relu",
                                     name="conv2d_"+str(n_gram))(x)

            # this output (n_filters x max_doc_len x 1)
            one_max = MaxPooling2D(pool_size=(1, self.preprocessor.max_sent_len - n_gram + 1), 
                                   name="pooling_"+str(n_gram))(cur_conv)

            # flip around, to get (1 x max_doc_len x n_filters)
            permuted = Permute((2,1,3), name="permuted_"+str(n_gram)) (one_max)
            
            # drop extra dimension
            r = Reshape((self.preprocessor.max_doc_len, self.n_filters), 
                            name="conv_"+str(n_gram))(permuted)
            
            convolutions.append(r)

        #sent_vectors = merge(convolutions, name="sentence_vectors", mode="concat")
        sent_vectors = concatenate(convolutions, name="sentence_vectors")
        sent_vectors = Dropout(self.sent_dropout, name="dropout")(sent_vectors)

        '''
        For this model, we simply take an unweighted sum of the sentence vectors
        to induce a document representation.
        ''' 
        def sum_sentence_vectors(x):
            return K.sum(x, axis=1)

        def sum_sentence_vector_output_shape(input_shape): 
            # should be (batch x max_doc_len x sentence_dim)
            shape = list(input_shape) 
            # something like (None, 96), where 96 is the
            # length of induced sentence vectors
            return (shape[0], shape[-1])
            
        doc_vector = Lambda(sum_sentence_vectors, 
                                output_shape=sum_sentence_vector_output_shape,
                                name="document_vector")(sent_vectors)

        doc_vector = Dropout(self.doc_dropout, name="doc_v_dropout")(doc_vector)
        output = Dense(1, activation="sigmoid", name="doc_prediction")(doc_vector)

        self.doc_model = Model(inputs=tokens_input, outputs=output)

        self.doc_model.compile(metrics=["accuracy",     
                                        RationaleCNN.metric_func_maker(metric_name="f", beta=self.f_beta), 
                                        RationaleCNN.metric_func_maker(metric_name="recall"), 
                                        RationaleCNN.metric_func_maker(metric_name="precision")], 
                                        loss="binary_crossentropy", optimizer="adadelta")
        print("doc-CNN model summary:")
        print(self.doc_model.summary())


    def build_RA_CNN_model(self):
        # input dim is (max_doc_len x max_sent_len) -- eliding the batch size
        tokens_input = Input(name='input', 
                            shape=(self.preprocessor.max_doc_len, self.preprocessor.max_sent_len), 
                            dtype='int32')
        

        # flatten; create a very wide matrix to hand to embedding layer
        tokens_reshaped = Reshape([self.preprocessor.max_doc_len*self.preprocessor.max_sent_len])(tokens_input)
        # embed the tokens; output will be (p.max_doc_len*p.max_sent_len x embedding_dims)
        # here we should initialize with weights from sentence model embedding layer!
        # also pass weights for initialization
        x = Embedding(self.preprocessor.max_features+1, self.preprocessor.embedding_dims, 
                        name="embedding")(tokens_reshaped)


        # reshape to preserve document structure -> 
        #       (doc_len x (word_in_sent x embedding_dim))

        # the 1 here is a dummy for the `channels' expected
        # by conv2d --> 
        #   (batch, channels, doc_len, (word_in_sent x embedding_dim))
        x = Reshape((1, self.preprocessor.max_doc_len, 
                     self.preprocessor.max_sent_len*self.preprocessor.embedding_dims), 
                     name="reshape")(x)
        #x = Reshape((self.preprocessor.max_doc_len, 
        #             self.preprocessor.max_sent_len*self.preprocessor.embedding_dims), 
        #             name="reshape")(x)

        total_sentence_dims = len(self.ngram_filters) * self.n_filters 

        convolutions = []
        for n_gram in self.ngram_filters:
            
            #cur_conv = Convolution2D(self.n_filters, 1, 
            #                         n_gram*self.preprocessor.embedding_dims, 
            #                         subsample=(1, self.preprocessor.embedding_dims),
            #                         activation="relu",
            #                         name="conv2d_"+str(n_gram))(x)

            cur_conv = Conv2D(self.n_filters, (1, n_gram*self.preprocessor.embedding_dims), 
                                strides=(1, self.preprocessor.embedding_dims),
                                name="conv2d_"+str(n_gram), activation="relu")(x)

            # this output (1 x new_rows x new_cols x n_filters)
            one_max = MaxPooling2D(pool_size=(1, self.preprocessor.max_sent_len-n_gram+1), 
                                   name="pooling_"+str(n_gram))(cur_conv)

            # flip around, to get (1 x max_doc_len x n_filters)
            permuted = Permute((2,1,3), name="permuted_"+str(n_gram)) (one_max)
            
            # drop extra dimension
            r = Reshape((self.preprocessor.max_doc_len, self.n_filters), 
                            name="conv_"+str(n_gram))(permuted)
            
            convolutions.append(r)

        sent_vectors = merge(convolutions, name="sentence_vectors", mode="concat")
        # it's not clear that it even makes sense to apply drop out here!
        #sent_vectors = Dropout(self.sent_dropout, name="dropout")(sent_vectors)

        # note that if end_to_end_train is False, we 'freeze' the sentence
        # softmax weights after pretraining the sentence model
        print("end-to-end training is: %s" % self.end_to_end_train)
        sent_pred_model = Dense(3, activation="softmax", name="sentence_prediction", kernel_regularizer=l2(0.01))
        sent_preds = TimeDistributed(sent_pred_model, name="sentence_predictions")(sent_vectors)

        ####
        # updating how we do sentence model 
        self.sentence_model = Model(inputs=tokens_input, outputs=sent_preds)
        
        self.sentence_model.compile(loss='categorical_crossentropy', 
                                    metrics=["accuracy"], 
                                    optimizer="adagrad")
        print (self.sentence_model.summary())
        
        #####

        
        sw_layer = Lambda(lambda x: K.max(x[:,0:2], axis=1), output_shape=(1,)) 
        
        # should really explicitly zero out sentences that were padded...
        sent_weights = TimeDistributed(sw_layer, name="sentence_weights")(sent_preds)
 
        def scale_merge(inputs):
            sent_vectors, sent_weights = inputs[0], inputs[1]
            return K.batch_dot(sent_vectors, sent_weights)

        def scale_merge_output_shape(input_shape):
            # this is expected now to be (None x sentence_vec_length x doc_length)
            # or, e.g., (None, 96, 200)
            input_shape_ls = list(input_shape)[0]
            # should be (batch x sentence embedding), e.g., (None, 96)
            return (input_shape_ls[0], input_shape_ls[1])


        # sent vectors will be, e.g., (None, 200, 96)
        # -> reshuffle for dot product below in merge -> (None, 96, 200)
        sent_vectors = Permute((2, 1), name="permuted_sent_vectors")(sent_vectors)

        # 8/10/2017 -- need to rework this into one of the new merge layers (dot?)
        #               this merge will eventually be deprecated apparently
        doc_vector = merge([sent_vectors, sent_weights], 
                                        name="doc_vector",
                                        mode=scale_merge,
                                        output_shape=scale_merge_output_shape)
        #doc_vector = 

        # trim extra dim
        doc_vector = Reshape((total_sentence_dims,), name="reshaped_doc")(doc_vector)
        doc_vector = Dropout(self.doc_dropout, name="doc_v_dropout")(doc_vector)

        doc_output = Dense(1, activation="sigmoid", name="doc_prediction")(doc_vector)
        
        
        # ... and compile
        self.doc_model = Model(inputs=tokens_input, outputs=doc_output)
        self.doc_model.compile(metrics=["accuracy", 
                                        RationaleCNN.metric_func_maker(metric_name="f"), 
                                        RationaleCNN.metric_func_maker(metric_name="recall"), 
                                        RationaleCNN.metric_func_maker(metric_name="precision")], 
                                loss="binary_crossentropy", optimizer="adam")

        self.set_final_sentence_model()

        print("rationale CNN model: ")
        print(self.doc_model.summary())


    def set_final_sentence_model(self):
        '''
        allow convenient access to sentence-level predictions, after training
        '''
        sent_prob_outputs = self.doc_model.get_layer("sentence_predictions")
        sent_model = K.function(inputs=self.doc_model.inputs + [K.learning_phase()], 
                        outputs=[sent_prob_outputs.output])
        self.sentence_prob_model = sent_model


    def predict_and_rank_sentences_for_doc(self, doc, num_rationales=3, return_rationale_indices=False, threshold=0):
        '''
        Given a Document instance, make doc-level prediction and return
        rationales.
        '''
        # @TODO making two preds seems awkward/inefficient!
        if self.sentence_prob_model is None:
            self.set_final_sentence_model()

        if doc.sentence_sequences is None:
            # this will be the usual case
            doc.generate_sequences(self.preprocessor)

        X_doc = np.array([doc.get_padded_sequences(self.preprocessor, labels_too=False)])
        
        # doc pred
        doc_pred = self.doc_model.predict(X_doc)[0][0]

        # now rank sentences; 0 indicates 'test time'
        sent_preds = self.sentence_prob_model(inputs=[X_doc, 0])[0].squeeze()[:doc.num_sentences]

        # bias_prob = 1 --> low risk 
        # recall: [1, 0, 0] -> positive rationale; [0, 1, 0] -> negative rationale
        idx = 0
        if doc_pred < .5:
            # pick neg rationales
            idx = 1

        rationale_indices = sent_preds[:,idx].argsort()[-num_rationales:]

        if return_rationale_indices:
            rationales = rationale_indices
        else:
            rationales = [doc.sentences[r_idx] for r_idx in rationale_indices]

        return (doc_pred, rationales)


    def train_sentence_model(self, train_documents, nb_epoch=5, 
                                downsample=True, 
                                sent_val_split=.2, 
                                sentence_model_weights_path="sentence_model_weights.hdf5"):

        # assumes sentence sequences have been generated!
        assert(train_documents[0].sentence_sequences is not None)

        # for the validation split, we assume this is at the *document*
        # level to be consistent with document-level training. 
        # so if this is .1, for example, the sentences comprising the last 
        # 10% of the documents will be used for validation
        validation_size = int(sent_val_split*len(train_documents))
        print("using sentences from %s docs for sentence prediction validation!" % 
                    validation_size)
    
        #######
        # build the train and validation sets
        # @TODO redundant blocks...
        ######
        X_doc, y_sent, train_sentences = [], [], []
        for d in train_documents[:-validation_size]:
            cur_X, cur_sent_y = d.get_padded_sequences(self.preprocessor)
            if np.max(cur_sent_y[:,:2]) > 0:            
                X_doc.append(cur_X)
                y_sent.append(cur_sent_y)
                train_sentences.append(d.padded_sentences)

        X_doc = np.array(X_doc)
        y_sent = np.array(y_sent)


        X_doc_validation, y_sent_validation, validation_sentences = [], [], []
        for d in train_documents[-validation_size:]:
            cur_X, cur_sent_y = d.get_padded_sequences(self.preprocessor)
            if np.max(cur_sent_y[:,:2]) > 0:
                # 12/13: only validate on samples that actually have at 
                # least one rationale!
                X_doc_validation.append(cur_X)
                y_sent_validation.append(cur_sent_y)
                validation_sentences.append(d.padded_sentences)
        X_doc_validation = np.array(X_doc_validation)
        y_sent_validation = np.array(y_sent_validation)



        if downsample:
            print("downsampling!")

            cur_acc, best_F, best_acc, best_loss = None, -np.inf, -np.inf, np.inf # - inf for F-score

            # then draw nb_epoch balanced samples; take one pass on each
            skip_count = 0
            for iter_ in range(nb_epoch):

                print ("on epoch: %s" % iter_)

                X_temp, y_sent_temp, sentences_temp = [], [], []
                for i in range(X_doc.shape[0]):
                    # i is indexing the document here!
                    X_doc_i = X_doc[i]
                    y_sent_i = y_sent[i]

                    # downsample each document
                        
                    '''
                    A tricky bit here is that the model expects a given doc length as input,
                    so here we take a kind of hacky approach of duplicating the downsampled
                    rows per documents. Basically this assembles 'balanced' pseudo documents
                    for input to the model.
                    '''
                    n_target_rows = X_doc_i.shape[0]
                    X_doc_i_temp, y_sent_i_temp, sampled_sentences = RationaleCNN.balanced_sample(X_doc_i, y_sent_i, 
                                                                                sentences=train_sentences[i],
                                                                                n_rows=n_target_rows)
                   
                    
                    X_temp.append(X_doc_i_temp)
                    y_sent_temp.append(y_sent_i_temp)
                    sentences_temp.append(sampled_sentences)

                X_temp = np.array(X_temp)
                y_sent_temp = np.array(y_sent_temp)
                
                self.sentence_model.fit(X_temp, y_sent_temp, epochs=1)

                cur_val_results = self.sentence_model.evaluate(X_doc_validation, y_sent_validation)
                
                out_str = ["%s: %s" % (metric, val) for metric, val in zip(self.sentence_model.metrics_names, cur_val_results)]
                print ("\n".join(out_str))

                

                loss, cur_acc = cur_val_results                
                if loss < best_loss:
                    best_acc = cur_acc
                    best_loss = loss 
                    self.sentence_model.save_weights(sentence_model_weights_path, overwrite=True)
                    print("new best sentence accuracy: %s\n" % best_acc)
                    print("new best sentence loss: %s\n" % best_loss)

  
        else:
            # using accuracy here because balanced(-ish) data is assumed.
            checkpointer = ModelCheckpoint(filepath=sentence_model_weights_path, 
                                    verbose=1,
                                    monitor="val_loss",
                                    save_best_only=True,
                                    mode="min")

            hist = self.sentence_model.fit(X_doc, y_sent, 
                        epochs=nb_epoch, 
                        validation_data=(X_doc_validation, y_sent_validation),
                        callbacks=[checkpointer])


        

        # reload best weights
        self.sentence_model.load_weights(sentence_model_weights_path)
        
        # 12/13/16 -- check if leaving sentence model trainable
        if not self.end_to_end_train:
            print ("freezing sentence prediction layer weights!")
            sent_softmax_layer = self.doc_model.get_layer("sentence_predictions")
            sent_softmax_layer.trainable = False 

            # after freezing these weights, recompile doc model (as per 
            # https://keras.io/getting-started/faq/#how-can-i-freeze-keras-layers)
            self.doc_model.compile(metrics=["accuracy",     
                                        RationaleCNN.metric_func_maker(metric_name="f", beta=self.f_beta), 
                                        RationaleCNN.metric_func_maker(metric_name="recall"), 
                                        RationaleCNN.metric_func_maker(metric_name="precision")], 
                                        loss="binary_crossentropy", optimizer="adadelta")

    def train_document_model(self, train_documents, nb_epoch=5, downsample=False, 
                                doc_val_split=.2, batch_size=50,
                                document_model_weights_path="document_model_weights.hdf5",
                                pos_class_weight=1):

        validation_size = int(doc_val_split*len(train_documents))
        print("validating using %s out of %s train documents." % (validation_size, len(train_documents)))

        ###
        # build the train set
        ###
        X_doc, y_doc = [], []
        y_sent = []
        for d in train_documents[:-validation_size]:
            cur_X, cur_sent_y = d.get_padded_sequences(self.preprocessor)
            X_doc.append(cur_X)
            y_doc.append(d.doc_y)
            y_sent.append(cur_sent_y)
        X_doc = np.array(X_doc)
        y_doc = np.array(y_doc)

        ####
        # @TODO refactor (rather redundant with above...)
        # and the validation set. 
        ####
        X_doc_validation, y_doc_validation = [], []
        y_sent_validation = []
        for d in train_documents[-validation_size:]:
            cur_X, cur_sent_y = d.get_padded_sequences(self.preprocessor)
            X_doc_validation.append(cur_X)
            y_doc_validation.append(d.doc_y)
            y_sent_validation.append(cur_sent_y)
        X_doc_validation = np.array(X_doc_validation)
        y_doc_validation = np.array(y_doc_validation)


        if downsample:
            print("downsampling!")

            cur_f, best_f = None, -np.inf  # - inf for F-score

            # then draw nb_epoch balanced samples; take one pass on each
            for iter_ in range(nb_epoch):

                print ("on epoch: %s" % iter_)

                X_tmp, y_tmp = RationaleCNN.balanced_sample(X_doc, y_doc, binary=True)

                self.doc_model.fit(X_tmp, y_tmp, batch_size=batch_size, epochs=1,
                                         class_weight={0:1, 1:pos_class_weight})

                cur_val_results = self.doc_model.evaluate(X_doc_validation, y_doc_validation)
                out_str = ["%s: %s" % (metric, val) for metric, val in zip(self.doc_model.metrics_names, cur_val_results)]
                print ("\n".join(out_str))

                loss, cur_acc, cur_f, cur_recall, cur_precision = cur_val_results                
                if cur_f > best_f:
                    best_f = cur_f
                    self.doc_model.save_weights(document_model_weights_path, overwrite=True)
                    print("new best F: %s\n" % best_f)


        else:
            # using accuracy here because balanced(-ish) data is assumed.
            checkpointer = ModelCheckpoint(filepath=document_model_weights_path, 
                                    verbose=1,
                                    monitor="val_acc",
                                    save_best_only=True,
                                    mode="max")


            hist = self.doc_model.fit(X_doc, y_doc, 
                        epochs=nb_epoch, 
                        validation_data=(X_doc_validation, y_doc_validation),
                        callbacks=[checkpointer],
                        batch_size=batch_size,
                        class_weight={0:1, 1:pos_class_weight})


        # reload best weights
        self.doc_model.load_weights(document_model_weights_path)

class Document:
    def __init__(self, doc_id, sentences, doc_label=None, sentences_labels=None, 
                    min_sent_len=1):
        self.doc_id = doc_id
        self.doc_y = doc_label

        self.sentences, self.sentences_y = [], []
        for idx, s in enumerate(sentences):
            if len(s.split(" ")) >= min_sent_len:
                self.sentences.append(s)
                if not sentences_labels is None:
                    self.sentences_y.append(sentences_labels[idx])

        self.sentence_sequences = None
        # length, pre-padding!
        self.num_sentences = len(self.sentences)

        self.sentence_weights = None 
        self.sentence_idx = 0
        self.n = len(self.sentences)


    def get_padded_sentences():
        # sometimes useful for indexing purposes
        return self.padded_sentences

    def __len__(self):
        return self.n 

    def generate_sequences(self, p):
        ''' 
        p is a preprocessor that has been instantiated
        elsewhere! this will be used to map sentences to 
        integer sequences here.
        '''
        self.sentence_sequences = p.build_sequences(self.sentences)
        self.padded_sentences = self.sentences + [''] * (p.max_doc_len - self.n)


    def get_padded_sequences_for_X_y(self, p, X, y):
        n_sentences = X.shape[0]
        if n_sentences > p.max_doc_len:
            X = X[:p.max_doc_len]
            y = y[:p.max_doc_len]
        elif n_sentences < p.max_doc_len:
            #dummy_rows = p.max_features * np.ones((p.max_doc_len-n_sentences, p.max_sent_len), dtype='int32') 
            dummy_rows = 0 * np.ones((p.max_doc_len-n_sentences, p.max_sent_len), dtype='int32')
            X = np.vstack((X, dummy_rows))
        
            dummy_lbls = [np.array([0,0,1]) for _ in range(p.max_doc_len-n_sentences)]
            y = np.vstack((y, dummy_lbls))

        return np.array(X), np.array(y)

    def get_padded_sequences_for_X(self, p, X):
        n_sentences = X.shape[0]
        if n_sentences > p.max_doc_len:
            X = X[:p.max_doc_len]
        elif n_sentences < p.max_doc_len:
            # pad
            #dummy_rows = p.max_features * np.ones((p.max_doc_len-n_sentences, p.max_sent_len), dtype='int32') 
            dummy_rows = 0 * np.ones((p.max_doc_len-n_sentences, p.max_sent_len), dtype='int32')
            X = np.vstack((X, dummy_rows))
        return np.array(X)


    def get_padded_sequences(self, p, labels_too=True):
        # return p.build_sequences(self.sentences, pad_documents=True)              
        #n_sentences = self.sentence_sequences.shape[0]
        X = self.sentence_sequences

        if labels_too:
            y = self.sentences_y
            return self.get_padded_sequences_for_X_y(p, X, y)

        # otherwise only return X
        return self.get_padded_sequences_for_X(p, X)

class Preprocessor:
    def __init__(self, max_features, max_sent_len, embedding_dims=200, wvs=None, 
                    max_doc_len=500, stopword=True):
        '''
        max_features: the upper bound to be placed on the vocabulary size.
        max_sent_len: the maximum length (in terms of tokens) of the instances/texts.
        embedding_dims: size of the token embeddings; over-ridden if pre-trained
                          vectors is provided (if wvs is not None).
        '''

        self.max_features = max_features  
        self.tokenizer = Tokenizer(num_words=self.max_features)#num_words=self.max_features)
        self.max_sent_len = max_sent_len  # the max sentence length! @TODO rename; this is confusing. 
        self.max_doc_len = max_doc_len # w.r.t. number of sentences!

        self.use_pretrained_embeddings = False 
        self.init_vectors = None 
        if wvs is None:
            self.embedding_dims = embedding_dims
        else:
            # note that these are only for initialization;
            # they will be tuned!
            self.use_pretrained_embeddings = True
            # for new gensim format
            self.embedding_dims = wvs.syn0.shape[1] #wvs.vector_size
            self.word_embeddings = wvs

        
        self.stopword = stopword
        # lifted directly from spacy's EN list
        #self.stopwords = [u'all', u'six', u'just', u'less', u'being', u'indeed', u'over', u'move', u'anyway', u'four', u'not', u'own', u'through', u'using', u'fify', u'where', u'mill', u'only', u'find', u'before', u'one', u'whose', u'system', u'how', u'somewhere', u'much', u'thick', u'show', u'had', u'enough', u'should', u'to', u'must', u'whom', u'seeming', u'yourselves', u'under', u'ours', u'two', u'has', u'might', u'thereafter', u'latterly', u'do', u'them', u'his', u'around', u'than', u'get', u'very', u'de', u'none', u'cannot', u'every', u'un', u'they', u'front', u'during', u'thus', u'now', u'him', u'nor', u'name', u'regarding', u'several', u'hereafter', u'did', u'always', u'who', u'didn', u'whither', u'this', u'someone', u'either', u'each', u'become', u'thereupon', u'sometime', u'side', u'towards', u'therein', u'twelve', u'because', u'often', u'ten', u'our', u'doing', u'km', u'eg', u'some', u'back', u'used', u'up', u'go', u'namely', u'computer', u'are', u'further', u'beyond', u'ourselves', u'yet', u'out', u'even', u'will', u'what', u'still', u'for', u'bottom', u'mine', u'since', u'please', u'forty', u'per', u'its', u'everything', u'behind', u'does', u'various', u'above', u'between', u'it', u'neither', u'seemed', u'ever', u'across', u'she', u'somehow', u'be', u'we', u'full', u'never', u'sixty', u'however', u'here', u'otherwise', u'were', u'whereupon', u'nowhere', u'although', u'found', u'alone', u're', u'along', u'quite', u'fifteen', u'by', u'both', u'about', u'last', u'would', u'anything', u'via', u'many', u'could', u'thence', u'put', u'against', u'keep', u'etc', u'amount', u'became', u'ltd', u'hence', u'onto', u'or', u'con', u'among', u'already', u'co', u'afterwards', u'formerly', u'within', u'seems', u'into', u'others', u'while', u'whatever', u'except', u'down', u'hers', u'everyone', u'done', u'least', u'another', u'whoever', u'moreover', u'couldnt', u'throughout', u'anyhow', u'yourself', u'three', u'from', u'her', u'few', u'together', u'top', u'there', u'due', u'been', u'next', u'anyone', u'eleven', u'cry', u'call', u'therefore', u'interest', u'then', u'thru', u'themselves', u'hundred', u'really', u'sincere', u'empty', u'more', u'himself', u'elsewhere', u'mostly', u'on', u'fire', u'am', u'becoming', u'hereby', u'amongst', u'else', u'part', u'everywhere', u'too', u'kg', u'herself', u'former', u'those', u'he', u'me', u'myself', u'made', u'twenty', u'these', u'was', u'bill', u'cant', u'us', u'until', u'besides', u'nevertheless', u'below', u'anywhere', u'nine', u'can', u'whether', u'of', u'your', u'toward', u'my', u'say', u'something', u'and', u'whereafter', u'whenever', u'give', u'almost', u'wherever', u'is', u'describe', u'beforehand', u'herein', u'doesn', u'an', u'as', u'itself', u'at', u'have', u'in', u'seem', u'whence', u'ie', u'any', u'fill', u'again', u'hasnt', u'inc', u'thereby', u'thin', u'no', u'perhaps', u'latter', u'meanwhile', u'when', u'detail', u'same', u'wherein', u'beside', u'also', u'that', u'other', u'take', u'which', u'becomes', u'you', u'if', u'nobody', u'unless', u'whereas', u'see', u'though', u'may', u'after', u'upon', u'most', u'hereupon', u'eight', u'but', u'serious', u'nothing', u'such', u'why', u'off', u'a', u'don', u'whereby', u'third', u'i', u'whole', u'noone', u'sometimes', u'well', u'amoungst', u'yours', u'their', u'rather', u'without', u'so', u'five', u'the', u'first', u'with', u'make', u'once']
        self.stopwords = ["a", "about", "again", "all", "almost", "also", "although", "always", "among", "an", "and", "another", "any", "are", "as", "at", "b", "be", "because", "been", "before", "being", "between", "both", "but", "by", "c", "can", "could", "did", "do", "d", "does", "each", "either", "enough", "etc", "f", "for", "from", "had", "has", "have", "here", "how", "h", "i", "if", "in", "into", "is", "it", "its", "j", "just", "k", "made", "make", "may", "must", "n", "o", "of", "often", "on", "p", "q", "r", "s", "so", "that", "the", "them", "then", "their", "those", "thus", "to", "t", "u", "use", "used", "v", "w", "x", "y", "z", "we", "was"]


    def remove_stopwords(self, texts):
        stopworded_texts = []
        for text in texts: 
            # note the naive segmentation; although this is same as the 
            # keras module does.
            #stopworded_text = " ".join([t for t in text.split(" ") if not t.lower() in self.stopwords])
            stopworded_text = []
            for t in text.split(" "):
                if not t in self.stopwords:
                    if t.isdigit():
                        t = "numbernumbernumber"
                    stopworded_text.append(t)
            #stopworded_text = " ".join([t for t in text.split(" ") if not t in self.stopwords])
            stopworded_text = " ".join(stopworded_text)
            stopworded_texts.append(stopworded_text)
        return stopworded_texts


    def preprocess(self, all_docs):
        ''' 
        This fits tokenizer and builds up input vectors (X) from the list 
        of texts in all_texts. Needs to be called before train!
        '''
        self.raw_texts = all_docs
        if self.stopword:
            #for text in self.raw_texts: 
            self.processed_texts = self.remove_stopwords(self.raw_texts)
        else:
            self.processed_texts = self.raw_texts

        self.fit_tokenizer()
        if self.use_pretrained_embeddings:
            self.init_word_vectors()


    def fit_tokenizer(self):
        ''' Fits tokenizer to all raw texts; remembers indices->words mappings. '''
        self.tokenizer.fit_on_texts(self.processed_texts)
        self.word_indices_to_words = {}
        for token, idx in self.tokenizer.word_index.items():
            self.word_indices_to_words[idx] = token


    def decode(self, x):
        ''' For convenience; map from word index vector to words'''
        words = []
        for t_idx in x:
            if t_idx == 0:
                words.append("pad")
            else: 
                words.append(self.word_indices_to_words[t_idx])
        return " ".join(words) 

    def build_sequences(self, texts, pad_documents=False):
        processed_texts = texts 
        if self.stopword:
            processed_texts = self.remove_stopwords(texts)

        X = list(self.tokenizer.texts_to_sequences_generator(processed_texts))

        # need to pad the number of sentences, too.
        X = np.array(pad_sequences(X, maxlen=self.max_sent_len))

        return X

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

        # init padding token!
        self.init_vectors.append(np.zeros(self.embedding_dims))

        # note that we make this a singleton list because that's
        # what Keras wants. 
        self.init_vectors = [np.vstack(self.init_vectors)]
