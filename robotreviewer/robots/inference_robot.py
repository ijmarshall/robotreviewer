

import json, requests

URL = 'http://trialstreamer.ccs.neu.edu'
PORT = 8000


class InferenceRobot:

    def __init__(self):
        pass 

    def annotate(self, articles_json):
        
        # get ev snippets 
        endpoint = '{}:{}/{}'.format(URL, PORT, 'get_ev')
        per_doc_ev = requests.post(url=endpoint, data=json.dumps(articles_json))
        for doc, ev in zip(articles_json, per_doc_ev.json()):
            doc['ev'] = ev

        # get_icos: returns a list of (i, c, o, label) tuples for each doc
        endpoint = '{}:{}/{}'.format(URL, PORT, 'get_icos')
        per_doc_icos = requests.post(url=endpoint, data=json.dumps(articles_json))
        for doc, icos in zip(articles_json, per_doc_icos.json()):
            doc['icos'] = icos

        return articles_json