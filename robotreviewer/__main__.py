"""
RobotReviewer server
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
import hashlib

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

# TODO: need to make sure directory is there on new install
rr_sql_conn = sqlite3.connect(robotreviewer.get_data('uploaded_pdfs/uploaded_pdfs.sqlite'), detect_types=sqlite3.PARSE_DECLTYPES)
c = rr_sql_conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS article(id INTEGER PRIMARY KEY, session_id TEXT, pdf_hash TEXT, pdf_file BLOB, annotations TEXT, timestamp TIMESTAMP)')
c.close()
rr_sql_conn.commit()

@app.route('/')
def main():
    return redirect('/upload_pdfs')

@app.route('/upload_pdfs')
def dropzone():
    # create new unique user ID (for the demo)
    robotreviewer_session_id = uuid.uuid4().hex
    resp = make_response(render_template('index.html'))
    resp.set_cookie('robotreviewer_session_id', robotreviewer_session_id)
    return resp
    

@app.route('/pdfview', methods=['GET'])
def pdfviewer():
    # processes the pdf view for individual study/annotation types
    robotreviewer_session_id = request.cookies['robotreviewer_session_id']
    db_id = request.args["study_id"]
    annotation_type = request.args["annotation_type"]
    c = rr_sql_conn.cursor()
    c.execute("SELECT pdf_file, annotations FROM article WHERE session_id=? AND id=?", (robotreviewer_session_id, db_id)) # each row_id should be unique; but to ensure that it is the correct session holder retrieving this data
    pdf_file, annotation_json = c.fetchone()
    data = MultiDict()
    data.load_json(annotation_json)
    marginalia = bots[annotation_type].get_marginalia(data)
    return json.dumps(marginalia)
    # return render_template('pdfview.html')

@csrf.exempt
@app.route('/add_pdfs_to_db', methods=['POST'])
def file_upload():
    robotreviewer_session_id = request.cookies['robotreviewer_session_id']    
    c = rr_sql_conn.cursor()
    for i, f in enumerate(request.files):
        blob = request.files[f].read()
        pdf_hash = hashlib.md5(blob).hexdigest()
        c.execute("INSERT INTO article (session_id, pdf_hash, pdf_file, timestamp) VALUES(?, ?, ?, ?)", [robotreviewer_session_id, pdf_hash, sqlite3.Binary(blob), datetime.now()])
        rr_sql_conn.commit()
    c.close()
    return "OK!"
    
@csrf.exempt # TODO: add csrf back in
@app.route('/synthesize_uploaded', methods=['POST'])
def synthesize_pdfs():
    # synthesise all PDFs uploaded with the same UID
    robotreviewer_session_id = request.cookies['robotreviewer_session_id']
    c = rr_sql_conn.cursor()
    articles = []
    for i, row in enumerate(c.execute("SELECT id, pdf_file FROM article WHERE session_id=?", (robotreviewer_session_id,))):
        print "ID number.. {}".format(row[0])
        data = pdf_reader.convert(row[1])    
        data = annotate(data, bot_names=["pubmed_bot", "bias_bot", "pico_bot", "rct_bot"])
        data.gold['db_id'] = row[0]
        articles.append(data)
    # here - save the annotations in the database as json
    for article in articles:
        c.execute("UPDATE article SET annotations = ? WHERE id = ?", (article.to_json(), article['db_id']))
        rr_sql_conn.commit()
    c.close()
    return "OK!"

@csrf.exempt # TODO: add csrf back in
@app.route('/report_view')
def show_report():
    robotreviewer_session_id = request.cookies['robotreviewer_session_id']
    c = rr_sql_conn.cursor()
    articles = []
    for i, row in enumerate(c.execute("SELECT annotations FROM article WHERE session_id=?", (robotreviewer_session_id,))):
        data = MultiDict()
        data.load_json(row[0])
        articles.append(data)
    return render_template('reportview.html', headers=bots['bias_bot'].get_domains(), articles=articles)



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
