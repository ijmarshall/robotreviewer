"""
the PICORobot class takes the title and abstract of a clinical trial as
input, and returns Population, Comparator/Intervention Outcome
information in the same format, which can easily be converted to JSON.

there are multiple ways to build a MultiDict, however the most common
way used in this project is as a PDF binary.

    pdf_binary = ...

    pdfr = PDFReader()
    data = pdfr.convert(pdf_binary)

    robot = PICORobot()
    annotations = robot.annotate(data)

The model was derived using the "Supervised Distant Supervision" strategy
introduced in our paper "Extracting PICO Sentences from Clinical Trial Reports
using Supervised Distant Supervision".
"""

# Authors:  Iain Marshall <mail@ijmarshall.com>
#           Joel Kuiper <me@joelkuiper.com>
#           Byron Wallace <byron@ccs.neu.edu>


import uuid
import logging

import numpy as np

import robotreviewer

# Needs amending
from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config



log = logging.getLogger(__name__)


class PICORobot:

    def __init__(self):
        """


        """
        logging.debug("Loading PICO LSTM-CRF")
        config = Config()

        #  ??following is needed
        data_prefix = "p1_all"
        cwd = os.getcwd() # need to use RR config/path
        config.filename_dev   = os.path.join(cwd, 'data', data_prefix + '_' + os.path.basename(config.filename_dev))
        config.filename_test  = os.path.join(cwd, 'data', data_prefix + '_' + os.path.basename(config.filename_test))
        config.filename_train = os.path.join(cwd, 'data', data_prefix + '_' + os.path.basename(config.filename_train))

        # build model
        self.model = NERModel(config)
        self.model.build()
        self.model.restore_session(config.dir_model)

        logging.debug("PICO classifiers loaded")




    def api_annotate(self, articles):

        if not all(('parsed_ab' in article for article in articles)):
            raise Exception('PICO span model requires an abstract to be able to complete annotation')

        
        annotations = []
        for article in articles:
            if article.get('skip_annotation'):
                annotations.append([])
            else:
                annotations.append(self.annotate(article['parsed_fullText']))

        
        
        return annotations


    def pdf_annotate(self, data):


        structured_data = self.annotate(doc_text)
        data.ml["pico_span"] = structured_data
        return data




    def annotate(self, article):

        """
        Annotate abstract of clinical trial report
        """

        label_dict = {"1_p": "population", "1_i": "interventions", "1_o": "outcomes"}


        parsed_text = nlp(text)
    
        out = {"population": [],
               "interventions": [],
               "outcomes": []}
        
        for sent in parsed_text.sents:
            words = [w.text for w in sent]
            preds = model.predict(words)
            
            last_label = "N"
            span = []
            
            for w, p in zip(words, preds):
                
                if p != last_label and span:
                    out[label_dict[last_label]].append(' '.join(span).strip())
                    span = []
                    
                if p != "N":
                    span.append(w)

                last_label = p

            if last_label != "N":
                out[label_dict[last_label]].append(' '.join(span).strip())

        return out

        

def main():
    # Sample code to make this run
    import unidecode, codecs, pprint
    # Read in example input to the text string
    with codecs.open('tests/example.txt', 'r', 'ISO-8859-1') as f:
        text = f.read()

    # make a PICO robot, use it to make predictions
    robot = PICORobot()
    annotations = robot.annotate(text)

    print("EXAMPLE OUTPUT:")
    print()
    pprint.pprint(annotations)


if __name__ == '__main__':
    main()
