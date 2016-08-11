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
c.execute('CREATE TABLE IF NOT EXISTS article(id INTEGER PRIMARY KEY, report_uuid TEXT, pdf_uuid TEXT, pdf_hash TEXT, pdf_file BLOB, annotations TEXT, timestamp TIMESTAMP)')
c.close()
rr_sql_conn.commit()

@app.route('/')
def main():
    resp = make_response(render_template('index.html'))
    return resp

@csrf.exempt
@app.route('/upload_and_annotate', methods=['POST'])
def upload_and_annotate():
    # uploads a bunch of PDFs, do the RobotReviewer annotation
    # save PDFs + annotations to database
    # returns the report run uuid + list of article uuids
    report_uuid = uuid.uuid4().hex
    pdf_uuids = []
    
    uploaded_files = request.files.getlist("file")
    c = rr_sql_conn.cursor()

    for f in uploaded_files:
        blob = f.read()
        pdf_hash = hashlib.md5(blob).hexdigest()
        pdf_uuid = uuid.uuid4().hex
        pdf_uuids.append(pdf_uuid)
        data = pdf_reader.convert(blob)
        data = annotate(data, bot_names=["pubmed_bot", "bias_bot", "pico_bot", "rct_bot"])
        data.gold['pdf_uuid'] = pdf_uuid

        c.execute("INSERT INTO article (report_uuid, pdf_uuid, pdf_hash, pdf_file, annotations, timestamp) VALUES(?, ?, ?, ?, ?, ?)", (report_uuid, pdf_uuid, pdf_hash, sqlite3.Binary(blob), data.to_json(), datetime.now()))
        rr_sql_conn.commit()
    c.close()

    return json.dumps({"report_uuid": report_uuid,
                       "pdf_uuids": pdf_uuids})
    # except:
    #     return "FAILED!"


@csrf.exempt # TODO: add csrf back in
@app.route('/report_view/<report_uuid>/<format>')
def show_report(report_uuid, format):    
    c = rr_sql_conn.cursor()
    articles, article_ids = [], []
    for i, row in enumerate(c.execute("SELECT pdf_uuid, annotations FROM article WHERE report_uuid=?", (report_uuid,))):
        data = MultiDict()
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

@app.route('/pdf/<report_uuid>/<pdf_uuid>')
def get_pdf(report_uuid, pdf_uuid):
    # returns PDF binary from database by pdf_uuid
    # where the report_uuid also matches
    c = rr_sql_conn.cursor()
    c.execute("SELECT pdf_file FROM article WHERE report_uuid=? AND pdf_uuid=?", (report_uuid, pdf_uuid)) # each row_id should be unique; but to ensure that it is the correct session holder retrieving this data
    pdf_file = c.fetchone()
    strIO = StringIO()
    strIO.write(pdf_file[0])
    strIO.seek(0)
    return send_file(strIO,
                     attachment_filename="filename=%s.pdf" % pdf_uuid,
                     as_attachment=False)


@app.route('/marginalia/<report_uuid>/<pdf_uuid>')
def get_marginalia(report_uuid, pdf_uuid):
    # calculates marginalia from database by pdf_uuid
    # where the report_uuid also matches
    annotation_type = request.args["annotation_type"]
    c = rr_sql_conn.cursor()
    c.execute("SELECT annotations FROM article WHERE report_uuid=? AND pdf_uuid=?", (report_uuid, pdf_uuid))
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
