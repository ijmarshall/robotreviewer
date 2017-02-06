"""
RobotReviewer server
"""

# Authors:  Iain Marshall <mail@ijmarshall.com>
#           Joel Kuiper <me@joelkuiper.com>
#           Byron Wallce <byron@ccs.neu.edu>

import logging, os
from datetime import datetime, timedelta

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

DEBUG_MODE = str2bool(os.environ.get("DEBUG", "true"))
LOCAL_PATH = "robotreviewer/uploads"
LOG_LEVEL = (logging.DEBUG if DEBUG_MODE else logging.INFO)
# determined empirically by Edward; covers 90% of abstracts
# (crudely and unscientifically adjusted for grobid)
NUM_WORDS_IN_ABSTRACT = 450

logging.basicConfig(level=LOG_LEVEL, format='[%(levelname)s] %(name)s %(asctime)s: %(message)s')
log = logging.getLogger(__name__)
log.info("Welcome to RobotReviewer :)")

from robotreviewer.textprocessing.pdfreader import PdfReader
pdf_reader = PdfReader() # launch Grobid process before anything else


from flask import Flask, json, make_response, send_file
from flask import redirect, url_for, jsonify
from flask import request, render_template

from werkzeug.utils import secure_filename

from flask_wtf.csrf import CsrfProtect
import zipfile

try:
    from cStringIO import StringIO # py2
except ImportError:
    from io import BytesIO as StringIO # py3

from robotreviewer.textprocessing.tokenizer import nlp

''' robots! '''
# from robotreviewer.robots.bias_robot import BiasRobot
from robotreviewer.robots.rationale_robot import BiasRobot
from robotreviewer.robots.pico_robot import PICORobot
from robotreviewer.robots.rct_robot import RCTRobot
from robotreviewer.robots.pubmed_robot import PubmedRobot
# from robotreviewer.robots.mendeley_robot import MendeleyRobot
# from robotreviewer.robots.ictrp_robot import ICTRPRobot
from robotreviewer.robots import pico_viz_robot
from robotreviewer.robots.pico_viz_robot import PICOVizRobot

from robotreviewer.data_structures import MultiDict
from robotreviewer import report_view

from robotreviewer import config
import robotreviewer

import uuid
from robotreviewer.util import rand_id
import sqlite3

import hashlib

import numpy as np # note - this should probably be moved!

app = Flask(__name__,  static_url_path='')
from robotreviewer import formatting
app.secret_key = os.environ.get("SECRET", "super secret key")
# setting max file upload size 100 mbs
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

csrf = CsrfProtect()
csrf.init_app(app)


######
## default annotation pipeline defined here
######
log.info("Loading the robots...")
bots = {"bias_bot": BiasRobot(top_k=3),
        "pico_bot": PICORobot(),
        "pubmed_bot": PubmedRobot(),
        # "ictrp_bot": ICTRPRobot(),
        "rct_bot": RCTRobot(),
        "pico_viz_bot": PICOVizRobot()}
        # "mendeley_bot": MendeleyRobot()}

log.info("Robots loaded successfully! Ready...")

#####
## connect to and set up database
#####
rr_sql_conn = sqlite3.connect(robotreviewer.get_data('uploaded_pdfs/uploaded_pdfs.sqlite'), detect_types=sqlite3.PARSE_DECLTYPES)
c = rr_sql_conn.cursor()

c.execute('CREATE TABLE IF NOT EXISTS article(id INTEGER PRIMARY KEY, report_uuid TEXT, pdf_uuid TEXT, pdf_hash TEXT, pdf_file BLOB, annotations TEXT, timestamp TIMESTAMP)')
c.close()
rr_sql_conn.commit()



# lastly wait until Grobid is connected
pdf_reader.connect()

@app.route('/')
def main():
    resp = make_response(render_template('index.html'))
    return resp

@csrf.exempt # TODO: add csrf back in
@app.route('/upload_and_annotate', methods=['POST'])
def upload_and_annotate():
    # uploads a bunch of PDFs, do the RobotReviewer annotation
    # save PDFs + annotations to database
    # returns the report run uuid + list of article uuids

    report_uuid = rand_id()
    pdf_uuids = []

    uploaded_files = request.files.getlist("file")
    c = rr_sql_conn.cursor()

    blobs = [f.read() for f in uploaded_files]
    filenames = [f.filename for f in uploaded_files]

    articles = pdf_reader.convert_batch(blobs)
    parsed_articles = []
    # tokenize full texts here
    for doc in nlp.pipe((d.get('text', u'') for d in articles), batch_size=1, n_threads=config.SPACY_THREADS, tag=True, parse=True, entity=False):
        parsed_articles.append(doc)


    # adjust the tag, parse, and entity values if these are needed later
    for article, parsed_text in zip(articles, parsed_articles):
        article._spacy['parsed_text'] = parsed_text

    for filename, blob, data in zip(filenames, blobs, articles):
        pdf_hash = hashlib.md5(blob).hexdigest()
        pdf_uuid = rand_id()
        pdf_uuids.append(pdf_uuid)
        data = annotate(data, bot_names=["bias_bot", "pico_bot", "rct_bot", "pico_viz_bot"])
        data.gold['pdf_uuid'] = pdf_uuid
        data.gold['filename'] = filename


        c.execute("INSERT INTO article (report_uuid, pdf_uuid, pdf_hash, pdf_file, annotations, timestamp) VALUES(?, ?, ?, ?, ?, ?)", (report_uuid, pdf_uuid, pdf_hash, sqlite3.Binary(blob), data.to_json(), datetime.now()))
        rr_sql_conn.commit()
    c.close()

    return json.dumps({"report_uuid": report_uuid,
                       "pdf_uuids": pdf_uuids})

@app.errorhandler(413)
def request_entity_too_large(error):
    ''' @TODO not sure if we want to return something else here? '''
    return json.dumps({'success':False, 'error':True}), 413, {'ContentType':'application/json'}


@csrf.exempt # TODO: add csrf back in
@app.route('/report_view/<report_uuid>/<format>')
def report_view(report_uuid, format):
    return produce_report(report_uuid, format, download=False)

@csrf.exempt # TODO: add csrf back in
@app.route('/download_report/<report_uuid>/<format>')
def download_report(report_uuid, format):
    report = produce_report(report_uuid, format, download=True)
    strIO = StringIO()
    strIO.write(report.encode('utf-8')) # need to send as a bytestring
    strIO.seek(0)
    return send_file(strIO,
                     attachment_filename="robotreviewer_report_%s.%s" % (report_uuid, format),
                     as_attachment=True)


# @TODO improve
# also should maybe be moved?
def get_study_name(article):
    authors = article.get("authors")
    study_str = ""
    if not authors is None:
        study_str = authors[0]["lastname"] + " et al."
    else: 
        #import pdb; pdb.set_trace()
        study_str = article['filename'][:20].lower().replace(".pdf", "") + " ..."
    return study_str


def produce_report(report_uuid, reportformat, download=False, PICO_vectors=True):
    c = rr_sql_conn.cursor()
    articles, article_ids = [], []
    error_messages = [] # accumulate any errors over articles
    for i, row in enumerate(c.execute("SELECT pdf_uuid, annotations FROM article WHERE report_uuid=?", (report_uuid,))):
        data = MultiDict()
        data.load_json(row[1])
        articles.append(data)
        article_ids.append(row[0])


    if reportformat=='html' or reportformat=='doc':
        # embeddings only relatively meaningful; do not generate
        # if we have only 1 study.
        if sum([(not article.get('_parse_error', False)) for article in articles]) < 2:
            # i.e. if we have fewer than 2 good articles then skip
            PICO_vectors = False 

        pico_plot_html = u""
        if PICO_vectors:
            study_names, p_vectors, i_vectors, o_vectors = [], [], [], []
            p_words, i_words, o_words = [], [], []
            for article in articles:
                if article.get('_parse_error'):
                    # need to make errors record more systematically
                    error_messages.append("{0}<br/>".format(get_study_name(article)))
                    
                else:    
                    study_names.append(get_study_name(article))
                    p_vectors.append(np.array(article.ml["p_vector"]))
                    p_words.append(article.ml["p_words"])

                    i_vectors.append(np.array(article.ml["i_vector"]))
                    i_words.append(article.ml["i_words"])

                    o_vectors.append(np.array(article.ml["o_vector"]))
                    o_words.append(article.ml["o_words"])


            vectors_d = {"population":np.vstack(p_vectors), 
                         "intervention":np.vstack(i_vectors), 
                         "outcomes":np.vstack(o_vectors)}

            words_d = {"population":p_words, "intervention":i_words, "outcomes":o_words}

            pico_plot_html = bots["pico_viz_bot"].generate_2d_viz(study_names, vectors_d, words_d,
                                            "{0}-PICO-embeddings".format(report_uuid))


        #import pdb; pdb.set_trace() 
        return render_template('reportview.{}'.format(reportformat), headers=bots['bias_bot'].get_domains(), articles=articles, 
                                pico_plot=pico_plot_html, report_uuid=report_uuid, online=(not download), 
                                errors=error_messages, reportformat=reportformat)
    elif reportformat=='json':
        return json.dumps({"article_ids": article_ids,
                           "article_data": [a.visible_data() for a in articles],
                           "report_id": report_uuid,
                           })
    else:
        raise Exception('format "{}" was requested but not available'.format(reportformat))

@app.route('/pdf/<report_uuid>/<pdf_uuid>')
def get_pdf(report_uuid, pdf_uuid):
    # returns PDF binary from database by pdf_uuid
    # where the report_uuid also matches
    c = rr_sql_conn.cursor()
    c.execute("SELECT pdf_file FROM article WHERE report_uuid=? AND pdf_uuid=?", (report_uuid, pdf_uuid))
    pdf_file = c.fetchone()
    strIO = StringIO()
    strIO.write(pdf_file[0])
    strIO.seek(0)
    return send_file(strIO,
                     attachment_filename="%s.pdf" % pdf_uuid,
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
    print(data.data)
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
    log.info('Cleaning up database')
    conn = sqlite3.connect(robotreviewer.get_data('uploaded_pdfs/uploaded_pdfs.sqlite'),
                           detect_types=sqlite3.PARSE_DECLTYPES)

    d = datetime.now() - timedelta(days=days)
    c = conn.cursor()
    c.execute("DELETE FROM article WHERE timestamp < datetime(?)", [d])
    conn.commit()
    conn.execute("VACUUM") # make the database smaller again
    conn.commit()
    conn.close()


try:
  from apscheduler.schedulers.background import BackgroundScheduler
  @app.before_first_request
  def initialize():
    log.info("initializing clean-up task")
    scheduler = BackgroundScheduler()
    scheduler.start()
    scheduler.add_job(cleanup_database, 'interval', hours=12)
except Exception:
  log.warn("could not start scheduled clean-up task")
