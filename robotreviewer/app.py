"""
RobotReviewer server
"""

# Authors:  Iain Marshall <mail@ijmarshall.com>
#           Joel Kuiper <me@joelkuiper.com>
#           Byron Wallace <byron@ccs.neu.edu>

import logging, os
from datetime import datetime, timedelta

import os
#os.environ["MKL_THREADING_LAYER"] = "GNU"

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

DEBUG_MODE = str2bool(os.environ.get("DEBUG", "true"))
LOCAL_PATH = "robotreviewer/uploads"
LOG_LEVEL = (logging.DEBUG if DEBUG_MODE else logging.INFO)

logging.basicConfig(level=LOG_LEVEL, format='[%(levelname)s] %(name)s %(asctime)s: %(message)s')
log = logging.getLogger(__name__)
log.info("Welcome to RobotReviewer server :)")

from flask import Flask, json, make_response, send_file
from flask import redirect, url_for, jsonify
from flask import request, render_template

from werkzeug.utils import secure_filename

from flask_wtf.csrf import CsrfProtect
import zipfile
import robotreviewer
from celery import Celery
from celery.result import AsyncResult

try:
    from cStringIO import StringIO # py2
except ImportError:
    from io import BytesIO as StringIO # py3
import uuid
from robotreviewer.util import rand_id
import sqlite3

''' robots! '''
# from robotreviewer.robots.bias_robot import BiasRobot
from robotreviewer.robots.rationale_robot import BiasRobot
from robotreviewer.robots.pico_robot import PICORobot
from robotreviewer.robots.rct_robot import RCTRobot
from robotreviewer.robots.pubmed_robot import PubmedRobot
# from robotreviewer.robots.mendeley_robot import MendeleyRobot
# from robotreviewer.robots.ictrp_robot import ICTRPRobot
#from robotreviewer.robots import pico_viz_robot
#from robotreviewer.robots.pico_viz_robot import PICOVizRobot
from robotreviewer.robots.sample_size_robot import SampleSizeBot

import hashlib

import numpy as np # note - this should probably be moved!

app = Flask(__name__,  static_url_path='')
from robotreviewer import formatting
from robotreviewer.data_structures import MultiDict
app.secret_key = os.environ.get("SECRET", "super secret key")
# setting max file upload size 100 mbs
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

csrf = CsrfProtect()
csrf.init_app(app)

#####
## connect to celery app
#####

celery_app = Celery('robotreviewer.ml_worker', backend='amqp://', broker='amqp://')
celery_tasks = {"annotate": celery_app.signature('robotreviewer.ml_worker.annotate')}

#####
## connect to database
#####
rr_sql_conn = sqlite3.connect(robotreviewer.get_data('uploaded_pdfs/uploaded_pdfs.sqlite'), detect_types=sqlite3.PARSE_DECLTYPES,  check_same_thread=False)



######
## default annotation pipeline defined here
## note we are using static methods, so not instantiating the classes
######
log.info("Loading the robots...")
bots = {"bias_bot": BiasRobot,
        "pico_bot": PICORobot,
        "pubmed_bot": PubmedRobot,
        # "ictrp_bot": ICTRPRobot(),
        "rct_bot": RCTRobot,
        #"pico_viz_bot": PICOVizRobot}#,
        "sample_size_bot":SampleSizeBot}
        # "mendeley_bot": MendeleyRobot()}

log.info("Robots loaded successfully! Ready...")
@app.route('/')
def main():
    resp = make_response(render_template('index.html'))
    return resp

@csrf.exempt # TODO: add csrf back in
@app.route('/upload_and_annotate', methods=['POST'])
def upload_and_annotate():
    # uploads a bunch of PDFs to the database,
    # then sends a Celery task for the
    # worker to do the RobotReviewer annotation
    # save PDFs + annotations to database
    # returns the report run uuid + list of article uuids

    report_uuid = rand_id()

    uploaded_files = request.files.getlist("file")
    c = rr_sql_conn.cursor()

    blobs = [f.read() for f in uploaded_files]
    pdf_hashes = [hashlib.md5(blob).hexdigest() for blob in blobs]
    filenames = [f.filename for f in uploaded_files]
    pdf_uuids = [rand_id() for fn in filenames]

    for pdf_uuid, pdf_hash, filename, blob in zip(pdf_uuids, pdf_hashes, filenames, blobs):
        c.execute("INSERT INTO doc_queue (report_uuid, pdf_uuid, pdf_hash, pdf_filename, pdf_file, timestamp) VALUES (?, ?, ?, ?, ?, ?)", (report_uuid, pdf_uuid, pdf_hash, filename, sqlite3.Binary(blob), datetime.now()))
        rr_sql_conn.commit()
    c.close()

    # send async request to Celery
    celery_tasks['annotate'].apply_async((report_uuid, ), task_id=report_uuid)

    return json.dumps({"report_uuid": report_uuid})

@csrf.exempt
@app.route('/annotate_status/<report_uuid>')
def annotate_status(report_uuid):
    '''
    check and return status of celery annotation process
    '''
    result = AsyncResult(report_uuid, app=celery_app)
    return json.dumps({"state": result.state, "meta": result.result})





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


def get_study_name(article):
    authors = article.get("authors")
    if authors:
        if len(authors) == 1:
            return authors[0]["lastname"] + ", " + \
                authors[0]["forename"] + " " + \
                authors[0]["initials"] + "."
        else:
            return authors[0]["lastname"] + " et al."
    else:
        return article['filename'][:20].lower().replace(".pdf", "") + " ..."


def produce_report(report_uuid, reportformat, download=False, PICO_vectors=False):
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
        '''
        PICO_vectors = False # just to avoid rendering for now TODO fix up
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
        '''

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
    import pdb; pdb.set_trace()
    annotations = annotation_pipeline(bot_names, data)
    return annotations

def annotation_pipeline(bot_names, data):
    for bot_name in bot_names:
        log.debug("Sending doc to {} for annotation...".format(bots[bot_name].__class__.__name__))

        data = bots[bot_name].annotate(data)
        log.debug("{} done!".format(bots[bot_name].__class__.__name__))
    return data

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
    c.execute("DELETE FROM article WHERE timestamp < datetime(?) AND dont_delete=0", [d])
    conn.commit()
    conn.execute("VACUUM") # make the database smaller again
    conn.commit()
    conn.close()


try:
    from apscheduler.schedulers.background import BackgroundScheduler

    @app.before_first_request
    def initialize():
        log.info("Initializing clean-up task")
        scheduler = BackgroundScheduler()
        scheduler.start()
        scheduler.add_job(cleanup_database, 'interval', hours=12)

except Exception:
    log.warn("Could not start scheduled clean-up task")
