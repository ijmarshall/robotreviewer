"""
RobotReviewer server
"""

# Authors:  Iain Marshall <mail@ijmarshall.com>
#           Joel Kuiper <me@joelkuiper.com>
#           Byron Wallce <byron.wallace@utexas.edu>

import os, logging
from flask import Flask, json, make_response, send_file
from flask import redirect, url_for, jsonify
from flask import request, render_template

from flask_wtf.csrf import CsrfProtect
import zipfile

try:
    from cStringIO import StringIO # py2
except ImportError:
    from io import StringIO # py3

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
if(DEBUG_MODE):
    app.run(debug=True)

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
c.execute('CREATE TABLE IF NOT EXISTS article(id INTEGER PRIMARY KEY, session_id TEXT, pdf_hash TEXT, pdf_uuid TEXT, pdf_file BLOB, annotations TEXT, timestamp TIMESTAMP)')
c.close()
rr_sql_conn.commit()

@app.route('/')
def main():
    # create new unique user ID (for the demo)
    robotreviewer_session_id = uuid.uuid4().hex
    resp = make_response(render_template('index.html'))
    resp.set_cookie('robotreviewer_session_id', robotreviewer_session_id)
    return resp

@csrf.exempt
@app.route('/add_pdfs_to_db', methods=['POST'])
def add_pdfs_to_db():
    # receives the PDFs and adds to DB
    robotreviewer_session_id = request.cookies['robotreviewer_session_id']
    c = rr_sql_conn.cursor()
    for i, f in enumerate(request.files):
        blob = request.files[f].read()
        pdf_hash = hashlib.md5(blob).hexdigest()
        c.execute("INSERT INTO article (session_id, pdf_uuid, pdf_hash, pdf_file, timestamp) VALUES(?, ?, ?, ?, ?)", [robotreviewer_session_id, uuid.uuid4().hex, pdf_hash, sqlite3.Binary(blob), datetime.now()])
        rr_sql_conn.commit()
    c.close()
    return "OK!"


@csrf.exempt # TODO: add csrf back in
@app.route('/synthesize_uploaded', methods=['POST'])
def synthesize_pdfs():
    # synthesise all PDFs uploaded with the same session ID
    robotreviewer_session_id = request.cookies['robotreviewer_session_id']
    c = rr_sql_conn.cursor()
    articles = []
    for i, row in enumerate(c.execute("SELECT pdf_uuid, pdf_file FROM article WHERE session_id=?", (robotreviewer_session_id,))):
        data = pdf_reader.convert(row[1])
        data = annotate(data, bot_names=["pubmed_bot", "bias_bot", "pico_bot", "rct_bot"])
        data.gold['pdf_uuid'] = row[0]
        articles.append(data)
    # here - save the annotations in the database as json
    for article in articles:
        c.execute("UPDATE article SET annotations = ? WHERE pdf_uuid = ?", (article.to_json(), article['pdf_uuid']))
        rr_sql_conn.commit()
    c.close()
    return "OK!"

@csrf.exempt # TODO: add csrf back in
@app.route('/report_view/<format>')
def show_report(format):
    robotreviewer_session_id = request.cookies['robotreviewer_session_id']
    c = rr_sql_conn.cursor()
    articles, article_ids = [], []
    for i, row in enumerate(c.execute("SELECT pdf_uuid, annotations FROM article WHERE session_id=?", (robotreviewer_session_id,))):
        data = MultiDict()
        print row
        data.load_json(row[1])
        articles.append(data)
        article_ids.append(row[0])
    if format=='html':
        return render_template('reportview.html', headers=bots['bias_bot'].get_domains(), articles=articles)
    elif format=='json':
        return json.dumps({"document_ids": article_ids,
                       "report": render_template('reportview.html', headers=bots['bias_bot'].get_domains(), articles=articles),
                        "report_id": uuid.uuid4().hex})
    else:
        raise Exception('format "{}" was requested but not available')

@app.route('/pdf/<pdf_uuid>')
def get_pdf(pdf_uuid):
    # returns PDF binary from database by pdf_uuid
    # where the session id also matches
    robotreviewer_session_id = request.cookies['robotreviewer_session_id']
    c = rr_sql_conn.cursor()
    c.execute("SELECT pdf_file FROM article WHERE session_id=? AND pdf_uuid=?", (robotreviewer_session_id, pdf_uuid)) # each row_id should be unique; but to ensure that it is the correct session holder retrieving this data
    pdf_file = c.fetchone()
    strIO = StringIO()
    strIO.write(pdf_file[0])
    strIO.seek(0)
    return send_file(strIO,
                     attachment_filename="filename=%s.pdf" % pdf_uuid,
                     as_attachment=False)


@app.route('/marginalia/<pdf_uuid>', methods=['GET'])
def get_marginalia(pdf_uuid):
    # calculates marginalia from database by pdf_uuid
    # where the session id also matches
    robotreviewer_session_id = request.cookies['robotreviewer_session_id']
    annotation_type = request.args["annotation_type"]
    c = rr_sql_conn.cursor()
    c.execute("SELECT annotations FROM article WHERE session_id=? AND pdf_uuid=?", (robotreviewer_session_id, pdf_uuid))
    annotation_json = c.fetchone()
    data = MultiDict()
    data.load_json(annotation_json[0])
    marginalia = bots[annotation_type].get_marginalia(data)
    return json.dumps(marginalia)


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
