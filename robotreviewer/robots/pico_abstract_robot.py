'''
The PicoAbstractBot consumes abstracts and extracts PICO elements from these.
'''

class PicoAbstractBot:

    def __init__(self):
        self.magic_threshold = 0.1 #TODO: What does this mean?

    #TODO Redundantly almost the same as sample size one.
    def annotate(self, data):
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
            # if pico_pred is not None:
            #     n, confidence = sample_pico_pred
            #     if confidence >= self.magic_threshold:
            #         sample_size_str = n

        data.ml["population"] = population_pred
        data.ml["intervention"] = intervention_pred
        data.ml["outcome"] = outcome_pred

        return data
