import logging

from fuzzywuzzy import fuzz
import numpy as np
from scipy.sparse import csr_matrix
import sqlite3
from sklearn.feature_extraction.text import HashingVectorizer

import robotreviewer


log = logging.getLogger(__name__)


class PubmedRobot:

    def __init__(self):
        raw_data = np.load(robotreviewer.get_data('pubmed/pubmed_title_hash_2016_07_24.npz'))
        self.vec_ti = csr_matrix((raw_data['data'], raw_data['indices'], raw_data['indptr']), raw_data['shape'])
        self.pmid_ind = np.load(robotreviewer.get_data('pubmed/pubmed_index_2016_07_24.npz'))['pmid_ind']
        self.vectorizer = HashingVectorizer(binary=True, stop_words='english')
        # load database
        self.connection = sqlite3.connect(robotreviewer.get_data('pubmed/pubmed_rcts_2016_07_24.sqlite'))
        self.c = self.connection.cursor()


    def pdf_annotate(self, data):

        title_text = data.get('title')
        if not title_text:
            log.error('Unable to run pubmed matching since we have no title')
            # unable to do pubmed unless we have a title, so just return the original data
            return data

        vec_q = self.vectorizer.transform([title_text])
        token_overlap = vec_q.dot(self.vec_ti.T)
        self.to = token_overlap
        best_ind = token_overlap.indices[token_overlap.data.argmax()]
        pmid = int(self.pmid_ind[best_ind])

        # checking both the overall similarity, and overlap similarity

        pubmed_data = self.query_pubmed(pmid)

        match_pc = fuzz.ratio(title_text.lower(), pubmed_data['title'].lower())
        match_pc_overlap = fuzz.partial_ratio(title_text.lower(), pubmed_data['title'].lower())

        # seems like a reasonable heuristic but not checked
        # (given that sometimes our query is a partial title
        # retrieved by Grobid)
        pubmed_data['pubmed_match_quality'] = sum([match_pc, match_pc_overlap])

        var_map = [('abstract', pubmed_data['abstract']),
                   ('pmid', pubmed_data['pmid']),
                   ('mesh', pubmed_data['mesh'])]


        if pubmed_data['pubmed_match_quality'] > 180:
            data.data['pubmed'] = pubmed_data # until setattr is worked out
        else:
             # keep it just in case, but don't replace better quality match
             data.data['dubious'] = pubmed_data # until setattr is worked out

        return data

    def query_pubmed(self, pmid):
        out = {}
        k_list = ["pmid", "title", "abstract", "year", "month", "volume", "issue", "pages", "journal", "journal_abbr"]
        self.c.execute("SELECT * FROM article WHERE pmid = ?", (pmid,))
        # TMP
        result = self.c.fetchone()
        out.update(zip(k_list, result))

        # pmid INTEGER, initials TEXT, forename TEXT, lastname TEXT
        k_list = ["pmid", "initials", "forename", "lastname"]
        self.c.execute("SELECT * FROM author WHERE pmid = ?", (pmid,))
        out["authors"] = [dict(zip(k_list, m)) for m in self.c.fetchall()]

        self.c.execute("SELECT * FROM mesh WHERE pmid = ?", (pmid,))
        out["mesh"] = [m[1] for m in  self.c.fetchall()]

        self.c.execute("SELECT * FROM ptyp WHERE pmid = ?", (pmid,))
        out["ptyp"] = [m[1] for m in  self.c.fetchall()]

        # self.c.execute("SELECT * FROM registry WHERE pmid = ?", (pmid,))
        # registry_ids = [m[1] for m in  self.c.fetchall()]
        #if registry_ids:
        #    out["registry"] = registry_ids # only store if these exist
        return out

    def short_citation(self, data):
        return u"{} {}, {}".format(data['authors'][0]['lastname'], data['authors'][0]['initials'], data['year'])



    @staticmethod
    def get_marginalia(data):
        """
        Get marginalia formatted for Spa from structured data
        """
        marginalia = [{"type": "PubMed",
            "title": "Title",
            "annotations": [],
            "description": data['title']}
        ]

        var_map = [('abstract', data['abstract']),
                   ('pmid', data['pmid']),
                   ('mesh', data['mesh'])]

        if data['pubmed_match_quality'] > 1.8:
            for k, v in var_map:
                if isinstance(v, list):
                    v_str = ', '.join(v)
                else:
                    v_str = unicode(v)
                marginalia.append ({"type": "PubMed",
                                  "title": k.capitalize(),
                                  "annotations": [],
                                  "description": v_str})
        else:
            for k, v in var_map:
                if isinstance(v, list):
                    v_str = ', '.join(v)
                else:
                    v_str = unicode(v)
                marginalia.append ({"type": "PubMed (*low quality match*)",
                                  "title": k.capitalize(),
                                  "annotations": [],
                                  "description": v_str})
        return marginalia
