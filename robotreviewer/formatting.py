"""
formatting.py
functions for displaying RobotReviewer internal data in useful ways
"""

from robotreviewer.app import app
import logging
log = logging.getLogger(__name__)


def format_authors(author_list, max_authors=1):
    et_al = False
    if len(author_list) > max_authors:
        et_al = True
        author_list = author_list[:max_authors]
    authors = u", ".join([u"{lastname} {initials}".format(**a) for a in author_list])
    if et_al:
        authors += " et al"
    return authors

@app.context_processor
def short_citation_fn():
    def short_citation(article):
        try:
            return u"{} {}, {}".format(article['authors'][0]['lastname'], article['authors'][0]['initials'], article.get('year', '[unknown year]'))
        except Exception as e:
            log.debug("Fallback: {} raised".format(e))
            return article['filename']
    return dict(short_citation=short_citation)

@app.context_processor
def long_citation_fn():
    def long_citation(article):
        try:
            bracket_issue = u"({})".format(article['issue']) if article.get('issue') else u""
            return u"{}. {} {} {}. {}{}; {}".format(format_authors(article['authors']), article['title'], article.get('journal_abbr', article['journal']), article.get('year', '[unknown year]'), article.get('volume', '?'), bracket_issue, article.get('pages', '?'))
        except Exception as e:
            log.debug("Fallback: {} raised".format(e))
            return u"Unable to extract citation information for file {}".format(article['filename'])
    return dict(long_citation=long_citation)

@app.context_processor
def not_rcts_fn():
    def not_rcts(articles):
        return [r for r in articles if r.get('rct', {}).get('is_rct', True) == False]
    return dict(not_rcts=not_rcts)


