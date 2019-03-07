'''
The SampleSizeBot consumes abstracts and extracts study sample sizes from these.
'''
import pickle



model_arch_path    = 'robotreviewer/data/sample_size/sample_size_model_architecture.json'
model_weights_path = 'robotreviewer/data/sample_size/sample_size_weights.hdf5'
preprocessor_path  = 'robotreviewer/data/sample_size/preprocessor.pickle'

class SampleSizeBot:

    def __init__(self):
        from robotreviewer.ml.sample_size_NN import MLPSampleSizeClassifier
        global MLPSampleSizeClassifier
        # as always, this was set in a totally and
        # completely scientific way.
        self.magic_threshold = 0.1

        with open(preprocessor_path, 'rb') as preprocessor_file:
            p = pickle.load(preprocessor_file)

        self.sample_size_model = MLPSampleSizeClassifier(p, model_arch_path, model_weights_path)
        

    def api_annotate(self, articles):

        if not all(('ab' in article for article in articles)):
            raise Exception('Sample size model requires abstract to be able to complete annotation')

        annotations = []

        for article in articles:
            if article.get('skip_annotation'):
                annotations.append({})
            else:
                sample_size_pred = self.sample_size_model.predict_for_abstract(article['ab'])
                if sample_size_pred is not None:
                    n, confidence = sample_size_pred
                    if confidence >= self.magic_threshold:
                        sample_size_str = n
                    else:
                        sample_size_str = 'not found'
                    annotations.append({"num_randomized": sample_size_str})
        return annotations


    def pdf_annotate(self, data):



        abstract = None

        if data.get("abstract") is not None:
            abstract = data["abstract"]
        elif data.get("parsed_text") is not None:
            # then just use the start of the document
            ABSTRACT_LEN = 420
            abstract = data['parsed_text'][:ABSTRACT_LEN].text

        sample_size_str = "???"
        if abstract is not None:
            sample_size_pred = self.sample_size_model.predict_for_abstract(abstract)
            if sample_size_pred is not None:
                n, confidence = sample_size_pred
                if confidence >= self.magic_threshold:
                    sample_size_str = n

        data.ml["sample_size"] = sample_size_str

        return data


    @staticmethod
    def get_marginalia(data):
        """
        Get marginalia formatted for Spa from structured data
        """
        marginalia = [{"type": "Sample size",
                      "title": "Sample size",
                      "annotations": [],
                      "description":  "Sample size: {0}".format(data["sample_size"])}]
        return marginalia

