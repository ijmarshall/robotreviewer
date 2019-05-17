import pickle
import string

import numpy as np

from nltk import word_tokenize
from nltk.corpus import stopwords

from keras.models import model_from_json
import keras.backend as K

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

import seaborn as sns
sns.set_style("white")
import mpld3

from matplotlib import rcParams
rcParams.update({'figure.subplot.left'  : 0.01})

import sys
from robotreviewer.ml import vectorizer
sys.modules['vectorizer'] = vectorizer


population_arch_path = 'robotreviewer/data/pico/PICO_embeddings/populations/architecture.json'
population_weight_path = 'robotreviewer/data/pico/PICO_embeddings/populations/weights.h5'
population_PCA_path = 'robotreviewer/data/pico/PICO_embeddings/populations/population-PCA.pickle'

intervention_arch_path = 'robotreviewer/data/pico/PICO_embeddings/interventions/architecture.json'
intervention_weight_path = 'robotreviewer/data/pico/PICO_embeddings/interventions/weights.h5'
intervention_PCA_path = 'robotreviewer/data/pico/PICO_embeddings/interventions/intervention-PCA.pickle'

outcomes_arch_path = 'robotreviewer/data/pico/PICO_embeddings/outcomes/architecture.json'
outcomes_weight_path = 'robotreviewer/data/pico/PICO_embeddings/outcomes/weights.h5'
outcomes_PCA_path = 'robotreviewer/data/pico/PICO_embeddings/outcomes/outcomes-PCA.pickle'

# shared across domains
vectorizer_path = 'robotreviewer/data/pico/PICO_embeddings/abstracts.p'


def convert_to_RGB(palette):
    RGB_palette = []
    for tuple in palette:
        RGB_palette.append("rgb(" + ",".join([str(int(255*v)) for v in tuple]) + ")")
    return RGB_palette

def scatter(study_names, X, ax, title="population", norm=False):
    n_studies = X.shape[0]
    palette = np.array(sns.color_palette("hls", n_studies))

    sc = ax.scatter(X[:,0], X[:,1], lw=0, s=60, c=palette[np.arange(n_studies)])


    # add labels for each study
    txts = []

    X_range = X[:,0].max()-X[:,0].min()
    Y_range = X[:,1].max()-X[:,1].min()

    for i in range(n_studies):
        # note that the below magic numbers are for aesthetics
        # only and are simply based on (manual) fiddling!
        jitter = np.array([-.05*X_range, .03*Y_range])
        xtext, ytext = X[i, :] + jitter
        txt = ax.text(xtext, ytext, study_names[i], fontsize=9)
        txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()])
        txts.append(txt)

    ax.set_xlim(min(X[:,0])-.5*X_range, max(X[:,0])+.5*X_range)
    ax.set_ylim(min(X[:,1])-.1*Y_range, max(X[:,1])+.1*Y_range)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    # also return palette for later use
    return sc, convert_to_RGB(palette[np.arange(n_studies)])


class PICOVizRobot:

    def __init__(self):

        self.elements = ["population", "intervention", "outcomes"]
        self.PCA_dict = {}
        self.load_models()

    def load_models(self):

        def _load_embedding_model(arch_path, weight_path):
            json_str = open(arch_path).read()
            model = model_from_json(json_str)
            model.load_weights(weight_path)

            inputs = [model.inputs[0], K.learning_phase()]

            # to trace back from embeddings to activations on n-grams
            # we provide intermediate output from conv filters here.
            outputs = [model.get_layer('convolution1d_1').output,
                       model.get_layer('convolution1d_2').output,
                       model.get_layer('convolution1d_3').output,
                       model.get_layer('study').output]

            return K.function(inputs, outputs)

        self.population_embedding_model = _load_embedding_model(population_arch_path,
                                                        population_weight_path)
        self.PCA_dict["population"] = pickle.load(open(population_PCA_path, 'rb'))

        self.intervention_embedding_model = _load_embedding_model(intervention_arch_path,
                                                        intervention_weight_path)
        self.PCA_dict["intervention"] = pickle.load(open(intervention_PCA_path, 'rb'))

        self.outcomes_embedding_model = _load_embedding_model(outcomes_arch_path,
                                                        outcomes_weight_path)
        self.PCA_dict["outcomes"] = pickle.load(open(outcomes_PCA_path, 'rb'))

        f = open(vectorizer_path, 'rb')
        self.vectorizer = pickle.load(f, encoding="latin")


    def generate_2d_viz(self, study_names, embeddings_dict, words_dict, name):

        # create 1 x 3 grid of plots (one per PICO element)
        f, axes_array = plt.subplots(1, 3)#, figsize=(15,30))

        # iterate over three embeddings (P/I/O)
        for i, element in enumerate(self.elements):
            X_hat = self.PCA_dict[element].transform(embeddings_dict[element])

            cur_ax = axes_array[i]

            points, RGB_palette = scatter(study_names, X_hat, cur_ax, title=element.lower())

            # setup labels; color code consisent w/scatter
            labels = []
            for study_idx, study_words in enumerate(words_dict[element]):
                label_str = u"<p style='color:{0}'>".format(RGB_palette[study_idx])
                label_str +=  ", ".join(study_words) + "</p>"
                labels.append(label_str)

            tooltip = mpld3.plugins.PointHTMLTooltip(points, labels=labels)
            mpld3.plugins.connect(f, tooltip)

        return mpld3.fig_to_html(f)

    def tokenize(self, text):
        tokenized = []
        for t in word_tokenize(text):
            t = t.lower()
            if any(char.isdigit() for char in t):
                t = "qqq"

            tokenized.append(t)

        return " ".join(tokenized)


    def postprocess_embedding(self, H):
        norms = np.apply_along_axis(np.linalg.norm, axis=1, arr=H)
        norms[norms == 0] = 1 # avoid division by zero norm
        H /= norms[:, np.newaxis]
        return H


    def get_activated_words(self, conv_output_uni, conv_output_bi, conv_output_tri, X, num_words=3):
        # conv_output will have dims (1 x abstract len x nb filters)
        # so something like (1 x 422 x 333)
        non_padding_idx = np.min(np.nonzero(X[0,:]))    # this is the first non-padding token index

        most_active_indices, activation_values = [], []

        # get most activated value for each n-gram spot, for each filter
        filter_maxes, filter_types, filter_indices = [], [], []
        for filter_idx, conv_output in enumerate([conv_output_uni, conv_output_bi, conv_output_tri]):
            filter_outputs = conv_output.squeeze() # drop the 1
            filter_outputs = filter_outputs[non_padding_idx:,:]
            # these are maxes over redundant filters corresponding to
            # a particular n-gram length (n \in {1,2,3})
            max_over_filters = np.amax(filter_outputs, axis=1)

            filter_maxes.extend(max_over_filters)
            filter_types.extend([filter_idx]*max_over_filters.shape[0])
            # for easy reverse indexing into X
            filter_indices.extend(list(range(max_over_filters.shape[0])))

        # now sort out most activated uni, bi, tri-grams.
        gram_indices = np.argsort(filter_maxes)[::-1]

        #most_active_indices.append(gram_indices)
        #    activation_values.append(max_over_filters[gram_indices])



        # 'qqq' is our 'out of vocab' string
        words_to_exclude = stopwords.words('english') + [t for t in string.punctuation] + ['qqq']
        def keep_word(word, already_observed=None):
            if already_observed is None:
                already_observed = []
            return not word in words_to_exclude and len(word)>1 and word not in already_observed

        ngrams = []
        for meta_idx in gram_indices:
            #index_vector = filter_indices[filter_types[idx]]
            idx = filter_indices[meta_idx]
            cur_n_gram = ""
            first_word = self.vectorizer.idx2word[X[0,non_padding_idx+idx]]
            if keep_word(first_word):
                cur_n_gram = first_word

            # collect the second token for bigrams, for example
            for offset in range(filter_types[meta_idx]):
                next_gram = self.vectorizer.idx2word[X[0,non_padding_idx+idx+offset+1]]
                if keep_word(next_gram):
                    cur_n_gram += " {0}".format(next_gram)


            if keep_word(cur_n_gram, already_observed=ngrams):
                ngrams.append(cur_n_gram)

        words = list(ngrams)[:num_words]
        return words

    def pdf_annotate(self, data):
        abstract, tokenized_abstract = "", ""


        if data.get("abstract") is not None:
            abstract = data["abstract"]
        elif data.get("parsed_text") is not None:
            # then just use the start of the document
            ABSTRACT_LEN = 420
            abstract = data['parsed_text'][:ABSTRACT_LEN].text
        else:
            # unable to annotate; return original data
            return data


        tokenized_abstract = self.tokenize(abstract)
        X = self.vectorizer.texts_to_sequences([tokenized_abstract])

        TEST_MODE = 0

        p_conv_output_uni, p_conv_output_bi, p_conv_output_tri, p_vec = self.population_embedding_model([X, TEST_MODE])
        p_vec = self.postprocess_embedding(p_vec)
        p_words = self.get_activated_words(p_conv_output_uni, p_conv_output_bi, p_conv_output_tri, X)

        i_conv_output_uni, i_conv_output_bi, i_conv_output_tri, i_vec = self.intervention_embedding_model([X, TEST_MODE])
        i_vec = self.postprocess_embedding(i_vec)
        i_words = self.get_activated_words(i_conv_output_uni, i_conv_output_bi, i_conv_output_tri, X)

        o_conv_output_uni, o_conv_output_bi, o_conv_output_tri, o_vec = self.outcomes_embedding_model([X, TEST_MODE])
        o_vec = self.postprocess_embedding(o_vec)
        o_words = self.get_activated_words(o_conv_output_uni, o_conv_output_bi, o_conv_output_tri, X)

        # we cast to a list because otherwise we cannot jsonify
        data.ml["p_vector"] = p_vec.tolist()
        data.ml["p_words"]  = p_words

        data.ml["i_vector"] = i_vec.tolist()
        data.ml["i_words"]  = i_words

        data.ml["o_vector"] = o_vec.tolist()
        data.ml["o_words"]  = o_words

        return data
