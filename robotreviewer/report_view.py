#
#   jinja template for the report view
#

from jinja2 import Template
import pandas as pd

# TODO
#  fix the rest of the defaults here
#  (problem is accessing [0] in an empty list - need to explicitly test + find defaults)

# + ALSO improve Grobid/pdf reader code since we're not currently accessing the authors there and Grobid *does* actually retrieve these



bias_domains = [u'Random sequence generation',
     u'Allocation concealment',
     u'Blinding of participants and personnel',
     u'Blinding of outcome assessment',
     u'Incomplete outcome data',
     u'Selective reporting']

def html(articles):
    t = Template(tm)
    return t.render(headers=bias_domains, articles=articles)

def csv(articles):
    column_headers = ["study ID", "domain", "judgement", "quote", "justification"]
    bias_table = []
    for article_i, article in enumerate(articles):
        for domain_i, domain in enumerate(article["bias"]):
            for sent_i, sent in enumerate(domain["justification"]):
                row = {}
                row["study ID"] = article["short_citation"] if domain_i==0 and sent_i==0 else ""
                row["judgement"] = domain["judgement"] if sent_i==0 else ""
                row["domain"] = domain["domain"] if sent_i==0 else ""
                row["quote"] = sent
                row["justification"] = ""
                bias_table.append(row)
            bias_table.append({}) # blank row
        bias_table.extend([{}, {}]) # two blank rows

    tab = pd.DataFrame(bias_table, columns=column_headers)
    return tab.to_csv(encoding='utf-8', index=False)


    