"""
the HumanRobot class takes the *abstract* of a clinical trial as
input as a string, and returns bias information as a dict which
can be easily converted to JSON.

A replica of the models described in the paper below:

@article{cohenProbabilisticAutomatedTagger2018,
  title = {A Probabilistic Automated Tagger to Identify Human-Related Publications},
  volume = {2018},
  issn = {1758-0463},
  journaltitle = {Database: The Journal of Biological Databases and Curation},
  urldate = {2019-08-22},
  author = {Cohen, Aaron M and Dunivin, Zackary O and Smalheiser, Neil R},
}

"""

# Authors:  Iain Marshall <mail@ijmarshall.com>
#           Joel Kuiper <me@joelkuiper.com>
#           Byron Wallce <byron.wallace@utexas.edu>

import json
import os
import robotreviewer
import pickle
import numpy as np

class HumanRobot:

    def __init__(self):
        """
        load the models
        """
        with open(os.path.join(robotreviewer.DATA_ROOT, "human/human_models.pck"), "rb") as f:
            self.human_models = pickle.load(f)



    def api_annotate(self, articles):

        """
        Annotate full text of clinical trial report
        `top_k` can be overridden here, else defaults to the class
        default set in __init__
        """

        if not all(((('ab' in article) and ('ti' in article)) or (article.get('skip_annotation')) for article in articles)):
            raise Exception('Human/non-human model requires a title and abstract to be able to complete annotation')

        fields = ["ti", "ab"]
        human_preds = []
        for field in fields:
            X = self.human_models["vecs"][field].transform((r[field] for r in articles))
            human_preds.append(self.human_models["clfs"][field].predict(X))

        human_preds = np.array(human_preds).T
        human_ens_preds = list(map(lambda x: {"is_human": bool(x)}, self.human_models["ensembler"].predict(human_preds)))
        return human_ens_preds
