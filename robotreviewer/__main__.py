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
from flask import Flask, json, make_response
from flask import redirect, url_for, jsonify
from flask import request, render_template

from flask_wtf.csrf import CsrfProtect
import zipfile

from robotreviewer.robots.bias_robot import BiasRobot
from robotreviewer.robots.pico_robot import PICORobot
from robotreviewer.robots.rct_robot import RCTRobot 
from robotreviewer.robots.pubmed_robot import PubmedRobot
from robotreviewer.data_structures import MultiDict

from robotreviewer import report_view 

from robotreviewer.textprocessing.pdfreader import PdfReader

from robotreviewer import config

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

DEBUG_MODE = str2bool(os.environ.get("DEBUG", "false"))

LOCAL_PATH = "robotreviewer/uploads"

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

bots = {"bias_bot": BiasRobot(top_k=3),
        "pico_bot": PICORobot(),
        "pubmed_bot": PubmedRobot(),
        "rct_bot": RCTRobot()}
log.info("Robots loaded successfully! Ready...")

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/pdfview')
def pdfviewer():
    return render_template('pdfview.html')


@csrf.exempt
@app.route('/file_upload', methods=['POST'])
def file_upload():
    print request.files
    for f in request.files:
        handler = request.files[f]
        filename = handler.filename
        path_to_file = os.path.join(LOCAL_PATH, filename)
        handler.save(path_to_file)
        return filename 

@csrf.exempt
@app.route('/annotate_abstract', methods=['POST'])
def is_rct_annotate():
    """
    processes JSON and returns for calls from web API
    """
    json_data = request.json
    annotations = annotate(json_data, bot_names=['bias_bot'])
    return json.dumps(annotations)
    


@csrf.exempt
@app.route('/synthesize_uploaded', methods=['GET', 'POST'])
def synthesize_pdfs():
    pdfs = request.form.lists()[0]
    assert(pdfs[0] == "pdfs[]")
    pdfs = pdfs[1] # list of pdf names.
    log.info("files to summarize: %s" % pdfs)
    return _generate_report_for_files(pdfs)

@csrf.exempt
@app.route('/generate_report', methods=['GET', 'POST'])
def generate_report():
    # TODO fix this into something better before making live
    # note, that the proper way to do this is via request.files

    # save as a temporary zip file...
    tmp_filename = os.path.join(config.TEMP_ZIP, 'tmp.zip')
    file = request.files['file']
    file.save(tmp_filename)

    return _generate_report_for_zip(tmp_filename)

def _generate_report_for_zip(filename):
    articles = []
    with zipfile.ZipFile(filename, "r") as f:
        for name in f.namelist():
            if name.startswith('__MACOSX/'):
                continue
            pdffilebin = f.read(name)
        
            tmp_filename = os.path.join(config.TEMP_PDF, 'tmp.pdf')
            with open(tmp_filename, 'wb') as tf:
                tf.write(pdffilebin)   

            data = pdf_reader.convert(tmp_filename)    
            data = annotate(data, bot_names=["pubmed_bot", "bias_bot", "pico_bot", "rct_bot"])
            articles.append(data)

    html = report_view.html(articles)
    response = make_response(html)
    response.headers["Content-Disposition"] = "attachment; filename=report.html"
    log.debug(html)
    return response
                    

# @TODO 
# this is an embarrassingly hacky method. the whole thing. sorry.
def _generate_report_for_files(files_list, MAX_ATTEMPTS=25):
    global pdf_reader  # lord forgive me
    articles = []
    for file_name in files_list:

        num_attempts = 0
        # as far as I can tell, grobid will periodically 
        # and stochastically fail on the same PDF. 
        # therefore, we simply try a bunch of times. 
        #
        # is this perhaps the best, most elegant "fix" ever?!?!?!
        while num_attempts < MAX_ATTEMPTS:
            try:
                full_filename = os.path.join(LOCAL_PATH, file_name)
                data = pdf_reader.convert(full_filename)    
                data = annotate(data, bot_names=["pubmed_bot", "bias_bot", "pico_bot", "rct_bot"])
                articles.append(data)
                break  
            except:
                log.info("failed on %s for a mysterious reason!" % file_name)
                log.info("on %s out of %s attempts" % (num_attempts+1, MAX_ATTEMPTS))
                pdf_reader.cleanup()
                pdf_reader = PdfReader() # re-init up Grobid connection
                num_attempts += 1
        


    html = report_view.html(articles)
    response = make_response(html)
    response.headers["Content-Disposition"] = "attachment; filename=report.html"
    return response


@app.route('/annotate_pdf', methods=['POST'])
def pdf_annotate():
    """
    API endpoint for PDF annotation from the Spa viewer
    Therefore returns only the processed annotations and
    not any of the structured data
    """
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
    data = annotate(data, bot_names=["pubmed_bot", "bias_bot", "pico_bot", "rct_bot"])
    return json.dumps({'marginalia': data['marginalia']})

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

def annotation_pipeline(bot_names, data):
    
    for bot_name in bot_names:
        log.debug("Sending doc to {} for annotation...".format(bots[bot_name].__class__.__name__))
        data = bots[bot_name].annotate(data)
        log.debug("{} done!".format(bots[bot_name].__class__.__name__))
    return data


if __name__ == '__main__':
    app.secret_key = 'super secret development key'
    # app.config['SESSION_TYPE'] = 'filesystem'

    app.run()
