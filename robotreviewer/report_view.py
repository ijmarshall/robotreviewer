#
#   jinja template for the report view
#

from jinja2 import Template
import pandas as pd

# TODO
#  fix the rest of the defaults here
#  (problem is accessing [0] in an empty list - need to explicitly test + find defaults)

# + ALSO improve Grobid/pdf reader code since we're not currently accessing the authors there and Grobid *does* actually retrieve these

tm = '''
<!DOCTYPE html>
<html>
    <head>
        <title>RobotReviewer report</title>
        <style>
            body {
                    font-family: Helvetica;
                    margin-left:auto; 
                    margin-right:auto;
                    width: 800px;
                }

            .bias-table {
                width: 400px;
                margin-left:auto; 
                margin-right:auto;
                padding: 0px;
                border-collapse: collapse;
                /*color: white;*/
            }

            .risk-low {
                background-color: #87BB2A;
                color: white;
                text-align: center;
                }

            .risk-high {
                background-color: #EA4747;
                color: white;
                text-align: center;
                }

            .trial-id-header {
                vertical-align: bottom;
                text-align: left;
            }

            th.rotate {
              /* Something you can count on */
              height: 300px;
              white-space: nowrap;
            }

            th.rotate > div {
              transform: translate(25px, 131px)
              rotate(315deg);
              width: 30px;
            }

            th.rotate > div > span {
              border-bottom: 1px solid #ccc;
              padding: 5px 10px;
            }


            table {
                   padding:20px;
                   margin: 20px 40px;

            }
        </style>
    </head>
    
    <body>

        <h1>RobotReviewer report</h1>
        
        <h2>Risk of bias table</h2>
        
        <table class='bias-table'>
            <tr class='bias-header-row'>
                <th class='trial-id-header'>trial</th>
                {% for header in headers %}
                    <th class="rotate"><div><span>{{ header }}</span></div></th>
                {% endfor %}
            </tr>
                
            
                {% for study in articles %}
                <tr>
                    <td class ="risk-studyid">{{ study['short_citation'] }}</td>
                    {% for domain in study['bias']['structured'] %}
                    <td class="{% if domain['judgement'] == 'low' %}risk-low{% elif domain['judgement'] == 'high/unclear' %}risk-high{% endif %}">{{ '+' if domain['judgement'] == 'low' else '?' }}</td>
                    {% endfor %}
                </tr>
                {% endfor %}
        </table>
        
        
        <h2>Characteristics of studies</h2>
        
        {% for study in articles %}
        <h3>{{ study['short_citation'] }}</h3>
        
        <table>
            {% for domain in study['pico_text']['structured'] %}            
            <tr>
                <td>{{ domain['domain'] }}</td><td>{{ domain['text'][0] }}</td>
                </tr>
                {% endfor %}
            
        
        <table>
            <tr> 
            <th>Bias</th><th>Judgement</th><th>Support for judgement</th>
            </tr>
            {% for domain in study['bias']['structured'] %}
            <tr>
                <td>{{ domain['domain'] }}</td><td>{{ domain['judgement'] }}</td><td>{{ domain['justification'][0] }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endfor %}
        
        <h2>References</h2>
        
        {% for study in articles %}
            <p>{{ loop.index }}. {{ study['citation'] }}</p>
        {% endfor %}
        
    </body>
</html>         
'''


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


    