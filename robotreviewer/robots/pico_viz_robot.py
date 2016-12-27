import pickle

import numpy as np 

from nltk import word_tokenize

import keras
from keras.models import model_from_json
import keras.backend as K

import sklearn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns

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


def rand_jitter(arr):
    stdev = 0.01*(arr.max()-arr.min())
    import pdb; pdb.set_trace()
    return arr + np.random.randn(arr.shape[0]) * stdev

def scatter(study_names, X, ax, title="population", norm=False):
    n_studies = X.shape[0] 
    palette = np.array(sns.color_palette("hls", n_studies))
    
    sc = ax.scatter(X[:,0], X[:,1], lw=0, s=60, c=palette[np.arange(n_studies)])

    ax.axis('tight')

    # add labels for each study
    txts = []
    #import pdb; pdb.set_trace()
    X_range = X[:,0].max()-X[:,0].min()
    Y_range = X[:,1].max()-X[:,1].min()

    for i in range(n_studies):   
        # note that the below magic numbers are for aesthetics
        # only and are simply based on (manual) fiddling!
        jitter = np.array([-.05*X_range, .03*Y_range]) 
        xtext, ytext = X[i, :] + jitter
        txt = ax.text(xtext, ytext, study_names[i], fontsize=8)
        txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()])
        txts.append(txt)

    ax.set_xlim(min(X[:,0])-.5*X_range, max(X[:,0])+.5*X_range)
    ax.set_ylim(min(X[:,1])-.1*Y_range, max(X[:,1])+.1*Y_range)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


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
            outputs = model.get_layer('study').output
                
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


    def generate_2d_viz(self, study_names, embeddings_dict, name): 

        # create 1 x 3 grid of plots (one per PICO element)
        f, axes_array = plt.subplots(1, 3)

        # iterate over three embeddings (P/I/O)
        for i, element in enumerate(self.elements):
            # first project

            X_hat = self.PCA_dict[element].transform(embeddings_dict[element])

            cur_ax = axes_array[i]
            scatter(study_names, X_hat, cur_ax, title=element.lower())

      
        outpath = "robotreviewer/static/img/RR_plots/{0}.png".format(name)
        plt.savefig(outpath)

        return "img/RR_plots/{0}.png".format(name)


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

    def annotate(self, data):
        '''
        df = pd.read_csv(csv_path)
        display(df.head())
        X = vectorizer.texts_to_sequences(df.text)
        TEST_MODE = 0

        H = embed([X, TEST_MODE])
        '''
        abstract, tokenized_abstract = "", ""

        
        if data.get("abstract") is not None:
            abstract = data["abstract"]
        elif data.get("parsed_text") is not None: 
            # then just use the start of the document
            ABSTRACT_LEN = 420 
            abstract = data['parsed_text'][:ABSTRACT_LEN].text
        else: 
            # @TODO what to do here? 
            pass 

        
        tokenized_abstract = self.tokenize(abstract)
        X = self.vectorizer.texts_to_sequences([tokenized_abstract])
        
        TEST_MODE = 0
        p_vec = self.postprocess_embedding(self.population_embedding_model([X, TEST_MODE]))
        i_vec = self.postprocess_embedding(self.intervention_embedding_model([X, TEST_MODE]))
        o_vec = self.postprocess_embedding(self.outcomes_embedding_model([X, TEST_MODE]))

        # we cast to a list because otherwise we cannot jsonify
        data.ml["p_vector"] = p_vec.tolist()
        data.ml["i_vector"] = i_vec.tolist()
        data.ml["o_vector"] = o_vec.tolist()
        return data
