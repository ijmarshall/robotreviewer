
from sklearn.feature_extraction.text import HashingVectorizer
import robotreviewer
import numpy as np
from scipy.sparse import csr_matrix
from fuzzywuzzy import process
from fuzzywuzzy import fuzz

import sqlite3
import os


class PubmedRobot:

    def __init__(self):
        raw_data = np.load(robotreviewer.get_data('pubmed/pubmed_title_hash_2016_07_24.npz'))
        self.vec_ti = csr_matrix((raw_data['data'], raw_data['indices'], raw_data['indptr']), raw_data['shape'])
        self.pmid_ind = np.load(robotreviewer.get_data('pubmed/pubmed_index_2016_07_24.npz'))['pmid_ind']
        self.vectorizer = HashingVectorizer(binary=True, stop_words='english')
        # load database
        self.connection = sqlite3.connect(robotreviewer.get_data('pubmed/pubmed_rcts_2016_07_24.sqlite'))
        self.c = self.connection.cursor()

    def annotate(self, data):

        title_text = data['title']
        if not title_text:
            # unable to do pubmed unless we have a title, so just return the original data
            return data

        vec_q = self.vectorizer.transform([title_text])
        token_overlap = vec_q.dot(self.vec_ti.T)
        self.to = token_overlap
        best_ind = token_overlap.indices[token_overlap.data.argmax()]
        pmid = self.pmid_ind[best_ind]

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

        pubmed_data['citation'] = self.format_citation(pubmed_data)
        pubmed_data['short_citation'] = self.short_citation(pubmed_data)
        
        if pubmed_data['pubmed_match_quality'] > 1.8:
            data.data['pubmed'] = pubmed_data # until setattr is worked out 
        else:
            # keep it just in case, but don't replace better quality match
            data.data['dubious'] = pubmed_data # until setattr is worked out

        return data

    def query_pubmed(self, pmid):
        out = {}
        k_list = ["pmid", "title", "abstract", "year", "month", "volume", "issue", "pages", "journal", "journal_abbr"]
        self.c.execute("SELECT * FROM article WHERE pmid = ?", (pmid,))    
        out.update(zip(k_list, self.c.fetchone()))
        
        # pmid INTEGER, initials TEXT, forename TEXT, lastname TEXT
        k_list = ["pmid", "initials", "forename", "lastname"]
        self.c.execute("SELECT * FROM author WHERE pmid = ?", (pmid,))
        out["authors"] = [dict(zip(k_list, m)) for m in self.c.fetchall()]
        
        self.c.execute("SELECT * FROM mesh WHERE pmid = ?", (pmid,))
        out["mesh"] = [m[1] for m in  self.c.fetchall()]
        
        self.c.execute("SELECT * FROM ptyp WHERE pmid = ?", (pmid,))
        out["ptyp"] = [m[1] for m in  self.c.fetchall()]
        return out

    def short_citation(self, data):
        return "{} {}, {}".format(data['authors'][0]['lastname'], data['authors'][0]['initials'], data['year'])

    def format_citation(self, data):
        bracket_issue = "({})".format(data['issue']) if data['issue'] else ""
        return u"{}. {} {} {}. {}{}; {}".format(self.format_authors(data['authors']), data['title'], data['journal_abbr'], data['year'], data['volume'], bracket_issue, data['pages'])


    def format_authors(self, author_list, max_authors=1):
        et_al = False
        if len(author_list) > max_authors:
            et_al = True
            author_list = author_list[:max_authors]
        authors = u", ".join([u"{lastname} {initials}".format(**a) for a in author_list])
        if et_al:
            authors += " et al"
        return authors

    @staticmethod
    def get_marginalia(data):
        """
        Get marginalia formatted for Spa from structured data
        """
        marginalia = [{"type": "PubMed",
            "title": "Citation",
            "annotations": [],
            "description": data['citation']}
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
    
