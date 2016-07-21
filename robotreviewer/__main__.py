"""
RobotReviewer server

Simple Flask server, which takes in the full text of a clinical
trial in JSON format, e.g.:

{"text": "Streptomycin Treatment of Pulmonary Tuberculosis: A Medical Research Council Investigation..."}

and outputs annotations in JSON format.

The JSON query should be sent as a POST query to:
`SERVER-NAME/annotate`
which by deafult would be localhost at:
`http://localhost:5000/annotate`
"""

# Authors:  Iain Marshall <mail@ijmarshall.com>
#           Joel Kuiper <me@joelkuiper.com>
#           Byron Wallce <byron.wallace@utexas.edu>

import os, logging
from flask import Flask, json
from flask import redirect, url_for, jsonify
from flask import request, render_template

from flask_wtf.csrf import CsrfProtect

from robotreviewer.robots.bias_robot import BiasRobot
from robotreviewer.robots.rct_robot import RCTRobot 

from robotreviewer.textprocessing.pdfreader import PdfReader

from robotreviewer import config

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

DEBUG_MODE = str2bool(os.environ.get("DEBUG", "false"))

# LOG_LEVEL = (logging.DEBUG if DEBUG_MODE else logging.INFO)
LOG_LEVEL = logging.DEBUG
logging.basicConfig(level=LOG_LEVEL, format='[%(levelname)s] %(name)s %(asctime)s: %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__,  static_url_path='')
csrf = CsrfProtect()
csrf.init_app(app)

app.debug = DEBUG_MODE

log.info("Welcome to RobotReviewer :)")



log.info("Loading the robots...")

pdf_reader = PdfReader() # set up Grobid connection

# nlp = English() is imported in __init__ to share among all

bots = {"bias_bot": BiasRobot(top_k=3)}
log.info("Robots loaded successfully! Ready...")

@app.route('/')
def main():
    return render_template('index.html')

@csrf.exempt
@app.route('/annotate_abstract', methods=['POST'])
def is_rct_annotate():
    """
    processes JSON and returns for calls from web API
    """
    json_data = request.json
    annotations = annotate(json_data, bot_names=['bias_bot'])
    return json.dumps(annotations)
    
@app.route('/generate_report', methods=['POST'])
def generate_report():
    # TODO fix this into something better before making live
    # note, that the proper way to do this is via request.files
    # There seems to be something being done unconventionally
    # in the Spa upload function, so that the upload is not
    # recognised as a file as it should be
    
    # save as a temporary zip file...
    tmp_filename = os.path.join(config.TEMP_ZIP, 'tmp.zip')

    with open(tmp_filename, 'wb') as f:
        f.write(request.data)

    annotations = annotate(data)

    return json.dumps(annotations)


@app.route('/annotate_pdf', methods=['POST'])
def pdf_annotate():
    # TODO fix this into something better before making live
    # note, that the proper way to do this is via request.files
    # There seems to be something being done unconventionally
    # in the Spa upload function, so that the upload is not
    # recognised as a file as it should be
    
    # save as a temporary PDF file...
    tmp_filename = os.path.join(config.TEMP_PDF, 'tmp.pdf')
    
    with open(tmp_filename, 'wb') as f:
        f.write(request.data)

    data = pdf_reader.convert(tmp_filename)
    annotations = annotate(data)

    return json.dumps(annotations)

@csrf.exempt
@app.route('/annotate', methods=['POST'])
def json_annotate():
    """
    processes JSON and returns for calls from web API
    """
    json_data = request.json
    annotations = annotate(json_data)
    return json.dumps(annotations)
    
    
def annotate(data, bot_names=["bias_bot"]):
    #
    # ANNOTATION TAKES PLACE HERE
    # change the line below if you wish to customise or
    # add a new annotator
    #
    annotations = annotation_pipeline(bot_names, data)
    return annotations



def annotation_pipeline(bot_names, text):
    output = {"marginalia": []}
    for bot_name in bot_names:
        log.debug("Sending doc to {} for annotation...".format(bots[bot_name].__class__.__name__))
        annotations = bots[bot_name].annotate(text)
        output["marginalia"].extend(annotations["marginalia"])
        log.debug("{} done!".format(bots[bot_name].__class__.__name__))
    return output


if __name__ == '__main__':
    app.secret_key = 'super secret development key'
    # app.config['SESSION_TYPE'] = 'filesystem'

    app.run()
