import logging

from mendeley import Mendeley
from robotreviewer import config

log = logging.getLogger(__name__)

class MendeleyRobot:

    def __init__(self):
        self.mendeley = Mendeley(config.MENDELEY_ID, config.MENDELEY_SECRET)
        self.mendeley_session = self.mendeley.start_client_credentials_flow().authenticate()


    def pdf_annotate(self, data):
        filehash = data.get('filehash')
        try:
            log.info('looking up PDF in Mendeley')
            doc = self.mendeley_session.catalog.by_identifier(filehash=filehash)

            log.info('title...')
            data.mendeley['title'] = doc.title
            log.info('year...')
            data.mendeley['year'] = doc.year
            log.info('abstract...')
            data.mendeley['abstract'] = doc.abstract
            log.info('authors...')
            data.mendeley['authors'] = [{'forename': a.first_name, 'lastname': a.last_name, 'initials': ''.join(name[0] for name in a.first_name.split())} for a in doc.authors]
        except:
            log.info('Unable to find PDF in Mendeley')

        return data





    # @staticmethod
    # def get_marginalia(data):
    #     """
    #     Get marginalia formatted for Spa from structured data
    #     """
    #     marginalia = [{"type": "PubMed",
    #         "title": "Title",
    #         "annotations": [],
    #         "description": data['title']}
    #     ]

    #     var_map = [('abstract', data['abstract']),
    #                ('pmid', data['pmid']),
    #                ('mesh', data['mesh'])]

    #     if data['pubmed_match_quality'] > 1.8:
    #         for k, v in var_map:
    #             if isinstance(v, list):
    #                 v_str = ', '.join(v)
    #             else:
    #                 v_str = unicode(v)
    #             marginalia.append ({"type": "PubMed",
    #                               "title": k.capitalize(),
    #                               "annotations": [],
    #                               "description": v_str})
    #     else:
    #         for k, v in var_map:
    #             if isinstance(v, list):
    #                 v_str = ', '.join(v)
    #             else:
    #                 v_str = unicode(v)
    #             marginalia.append ({"type": "PubMed (*low quality match*)",
    #                               "title": k.capitalize(),
    #                               "annotations": [],
    #                               "description": v_str})
    #     return marginalia
