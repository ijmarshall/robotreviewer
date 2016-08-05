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
import robotreviewer

import uuid
import sqlite3
from datetime import datetime

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

DEBUG_MODE = str2bool(os.environ.get("DEBUG", "false"))
LOCAL_PATH = "robotreviewer/uploads"

LOG_LEVEL = (logging.DEBUG if DEBUG_MODE else logging.INFO)
LOG_LEVEL = logging.DEBUG
logging.basicConfig(level=LOG_LEVEL, format='[%(levelname)s] %(name)s %(asctime)s: %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__,  static_url_path='')
csrf = CsrfProtect()
csrf.init_app(app)

log.info("Welcome to RobotReviewer :)")
log.info("Loading the robots...")

pdf_reader = PdfReader() # set up Grobid connection
# nlp = English() is imported in __init__ to share among all


######
## default annotation pipeline defined here
######
bots = {"bias_bot": BiasRobot(top_k=3),
        "pico_bot": PICORobot(),
        "pubmed_bot": PubmedRobot(),
        "rct_bot": RCTRobot()}
log.info("Robots loaded successfully! Ready...")

#####
## connect to and set up database 
#####
rr_sql_conn = sqlite3.connect(robotreviewer.get_data('uploaded_pdfs/uploaded_pdfs.sqlite'), detect_types=sqlite3.PARSE_DECLTYPES)
c = rr_sql_conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS article(user TEXT, pdf_file BLOB, timestamp TIMESTAMP)')
c.close()
rr_sql_conn.commit()

@app.route('/')
def main():
    # create new unique user ID (for the demo)
    robotreviewer_session_id = uuid.uuid4().hex
    resp = make_response(render_template('index.html'))
    resp.set_cookie('robotreviewer_session_id', robotreviewer_session_id)
    return resp
    

@app.route('/pdfview')
def pdfviewer():
    return render_template('pdfview.html')


@csrf.exempt
@app.route('/file_upload', methods=['POST'])
def file_upload():
    robotreviewer_session_id = request.cookies['robotreviewer_session_id']
    print "**** /file_upload received uuid {} ****".format(robotreviewer_session_id) # remove this later
    c = rr_sql_conn.cursor()
    for f in request.files:
        blob = request.files[f].read()
        c.execute("INSERT INTO article (user, pdf_file, timestamp) VALUES(?, ?, ?)", [robotreviewer_session_id, sqlite3.Binary(blob), datetime.now()])
        rr_sql_conn.commit()
    c.close()
    return "success"
    
@csrf.exempt # TODO: add csrf back in
@app.route('/synthesize_uploaded', methods=['POST'])
def synthesize_pdfs():
    # synthesise all PDFs uploaded with the same UID
    robotreviewer_session_id = request.cookies['robotreviewer_session_id']
    print "**** /synthesize_uploaded received uuid {} ****".format(robotreviewer_session_id) # remove this later
    return _generate_report_for_files(robotreviewer_session_id)


# @TODO 
# this is an embarrassingly hacky method. the whole thing. sorry.
def _generate_report_for_files(robotreviewer_session_id, MAX_ATTEMPTS=25):
    
    c = rr_sql_conn.cursor()
    
    

    # global pdf_reader  # lord forgive me

    articles = []
    
    for blob in c.execute("SELECT pdf_file FROM article WHERE user=?", (robotreviewer_session_id,)):

        num_attempts = 0
        # as far as I can tell, grobid will periodically 
        # and stochastically fail on the same PDF. 
        # therefore, we simply try a bunch of times. 
        #
        # is this perhaps the best, most elegant "fix" ever?!?!?!

        # while num_attempts < MAX_ATTEMPTS:
            # try:
        data = pdf_reader.convert(blob[0])    
        data = annotate(data, bot_names=["pubmed_bot", "bias_bot", "pico_bot", "rct_bot"])
        articles.append(data)
        # break  
            # except:
            #     log.info("failed on %s for a mysterious reason!" % file_name)
            #     log.info("on %s out of %s attempts" % (num_attempts+1, MAX_ATTEMPTS))
            #     pdf_reader.cleanup()
            #     pdf_reader = PdfReader() # re-init up Grobid connection
            #     num_attempts += 1
        
    c.close()
    html = report_view.html(articles)
    response = make_response(html)
    response.headers["Content-Disposition"] = "attachment; filename=report.html"
    return response



@csrf.exempt # TODO: add csrf back in
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


# TODO make something that calls this
def cleanup_database(days=1):
    """
    remove any PDFs which have been here for more than
    1 day, then compact the database
    """
    d = datetime.now() + timedelta(days=days)
    c = rr_sql_conn.cursor()
    c.execute("DELETE FROM article WHERE timestamp < datetime(?)", [d])
    rr_sql_conn.commit()
    rr_sql_conn.execute("VACUUM") # make the database smaller again


if __name__ == '__main__':
    app.secret_key = 'super secret development key'
    app.run()
