from robotreviewer.ml.pico_abstract_NN import PicoAbstractClassifier


import logging
log = logging.getLogger(__name__)

'''
The PicoAbstractBot consumes abstracts and extracts PICO elements from them
'''

class PicoAbstractBot:

    def __init__(self):
        self.magic_threshold = 0.1 #TODO: What does this mean?
        self.pico_model = PicoAbstractClassifier()
        self.pico_model.build_pico_model()

        log.debug("Created a PicoAbstactBot")

    #TODO Redundantly almost the same as sample size one.
    def annotate(self, data):
        log.debug("Annotated")
        abstract = None
        if data.get("abstract") is not None:
            abstract = data["abstract"]
        elif data.get("parsed_text") is not None:
            # then just use the start of the document
            ABSTRACT_LEN = 420
            abstract = data['parsed_text'][:ABSTRACT_LEN].text

        population_pred = "???"
        intervention_pred = "???"
        outcome_pred = "???"
        if abstract is not None:
            pico_pred = self.pico_model.predict_for_abstract(abstract)
            population_pred = pico_pred.population
            intervention_pred = pico_pred.intervention
            outcome_pred = pico_pred.outcome

            print(pico_pred)

        print(population_pred)
        print(intervention_pred)
        data.ml["population"] = population_pred
        data.ml["intervention"] = intervention_pred
        data.ml["outcome"] = outcome_pred

        return data
