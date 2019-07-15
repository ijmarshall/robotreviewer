import logging
log = logging.getLogger(__name__)
import numpy as np

model_arch_path    = 'robotreviewer/data/punchlines/punchline_model.json'
model_weights_path = 'robotreviewer/data/punchlines/punchline.weights.best.hdf5'


inf_model_arch_path    = 'robotreviewer/data/punchlines/inference_model.json'
inf_model_weights_path = 'robotreviewer/data/punchlines/inference.weights.best.hdf5'


class PunchlinesBot:

    def __init__(self):

        from robotreviewer.ml.punchline_extractor import PunchlineExtractor, SimpleInferenceNet

        global PunchlineExtractor
        global SimpleInferenceNet

        self.punchlines_model = PunchlineExtractor(architecture_path=model_arch_path, weights_path=model_weights_path)
        self.inference_model = SimpleInferenceNet(architecture_path=inf_model_arch_path, weights_path=inf_model_weights_path)


    def get_top_sentences(self, sentences, k=1):

        sentences_text = [s.text for s in sentences if s.text]

        if sentences_text == []:
            return []
        sentence_scores = self.punchlines_model.score_sentences(sentences_text).squeeze()
        sorted_indices = np.argsort(sentence_scores)[::-1]

        top_sentences = [sentences_text[idx] for idx in sorted_indices[:k]]

        return top_sentences

    def infer_result(self, sentence):
        direction_idx = np.argmax(self.inference_model.infer_result([sentence]))
        return ["↓ sig decrease", "― no diff", "↑ sig increase"][direction_idx]

    def annotate(self, data):
        pass

    def api_annotate(self, articles):

        if not all(('ab' in article for article in articles)):
            raise Exception('Punchline extraction model requires abstract to be able to complete annotation')


        out = []

        for article in articles:

            top_sentences = self.get_top_sentences(article['parsed_ab'].sents)
            if top_sentences:
                finding_direction = self.infer_result(top_sentences[0])
            else:
                finding_direction = "unable to determine"
            row = {"punchline_text": " ".join(top_sentences),
                   "effect": finding_direction}
            out.append(row)
        return out



    def pdf_annotate(self, data):
        log.info('retrieving text')
        doc_text = data.get('parsed_text')
        if not doc_text:
            return data # we've got to know the text at least..

        top_sentences = self.get_top_sentences(doc_text.sents)
        data.ml["punchlines"] = " ".join(top_sentences)

        finding_direction = self.infer_result(top_sentences[0])
        data.ml["finding_direction"] = finding_direction

        #return
        #structured_data.append({"domain":domain,
        #                        "text": high_prob_sents,

        log.info('extracted study punchlines; returning')
        #return {"domain":"Punchlines",
        #        "text": "; ".join(top_sentences),
        #        "annotations": []}

        return data




    @staticmethod
    def get_marginalia(data):

        marginalia = [{"type": "IAMHERE",
                      "title": "IAMHERE",
                      "annotations": [],
                      "description":  "HELLO"}]
        return marginalia
