#
#   jinja template for the report view
#

from jinja2 import Template


tm = '''
<!DOCTYPE html>
<html>
    <head>
        <title>RobotReviewer report</title>
        <link rel="stylesheet" type="text/css" href="robotreviewer-report.css">
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
                    {% for domain in study['bias'] %}
                    <td class="{% if domain['judgement'] == 'low' %}risk-low{% elif domain['judgement'] == 'high/unclear' %}risk-high{% endif %}">{{ '+' if domain['judgement'] == 'low' else '?' }}</td>
                    {% endfor %}
                </tr>
                {% endfor %}
        </table>
        
        
        <h2>Characteristics of studies</h2>
        
        {% for study in articles %}
        <h3>{{ study['authors'][0]['lastname'] }} {{ study['authors'][0]['initials'] }}, {{ study['year'] }} [{{ loop.index }}]</h3>
        
        <table>
            {% for domain in study['pico_text'] %}            
            <tr>
                <td>{{ domain['domain'] }}</td><td>{{ domain['text'][0] }}</td>
                </tr>
                {% endfor %}
            
        
        <table>
            <tr> 
            <th>Bias</th><th>Judgement</th><th>Support for judgement</th>
            </tr>
            {% for domain in study['bias'] %}
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

def compile(articles):
    t = Template(tm)
    return t.render(headers=bias_domains, articles=articles)

    