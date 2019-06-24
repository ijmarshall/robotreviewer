import logging
log = logging.getLogger(__name__)
import numpy as np

model_arch_path    = 'robotreviewer/data/punchlines/punchline_model.json'
model_weights_path = 'robotreviewer/data/punchlines/punchline.weights.best.hdf5'

class PunchlinesBot:

    def __init__(self):
        from robotreviewer.ml.punchline_extractor import PunchlineExtractor

        global PunchlineExtractor

        self.punchlines_model = PunchlineExtractor(architecture_path=model_arch_path, weights_path=model_weights_path)
       
    def get_top_sentences(self, sentences, k=1):
        
        sentences_text = [s.text for s in sentences]

        #import pdb; pdb.set_trace()

        #sentence_scores = np.array([self.punchlines_model.score_sentence(s) for s in sentences_text])
        sentence_scores = self.punchlines_model.score_sentences(sentences_text).squeeze()
        sorted_indices = np.argsort(sentence_scores)[::-1]

        top_sentences = [sentences_text[idx] for idx in sorted_indices[:k]]

        return top_sentences


    def annotate(self, data):
        pass 

    #def api_annotate(self, data):
    #    print("API annotate!")
        

    def pdf_annotate(self, data):
        log.info('retrieving text')
        doc_text = data.get('parsed_text')
        if not doc_text:
            return data # we've got to know the text at least..

        top_sentences = self.get_top_sentences(doc_text.sents)
        data.ml["punchlines"] = " ".join(top_sentences)

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