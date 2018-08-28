"""
RobotReviewer ML worker

called by `celery -A ml_worker worker --loglevel=info`

"""


# Authors:  Iain Marshall <mail@ijmarshall.com>
#           Joel Kuiper <me@joelkuiper.com>
#           Byron Wallace <byron@ccs.neu.edu>


from celery import Celery, current_task
from celery.contrib import rdb
from celery.signals import worker_init

import logging, os

import sqlite3
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

log.info("RobotReviewer machine learning tasks starting")


from robotreviewer.textprocessing.pdfreader import PdfReader
pdf_reader = PdfReader() # launch Grobid process before anything else


from robotreviewer.textprocessing.tokenizer import nlp

''' robots! '''
# from robotreviewer.robots.bias_robot import BiasRobot
from robotreviewer.robots.rationale_robot import BiasRobot
from robotreviewer.robots.pico_robot import PICORobot
from robotreviewer.robots.rct_robot import RCTRobot
from robotreviewer.robots.pubmed_robot import PubmedRobot
# from robotreviewer.robots.mendeley_robot import MendeleyRobot
# from robotreviewer.robots.ictrp_robot import ICTRPRobot
# from robotreviewer.robots import pico_viz_robot
# from robotreviewer.robots.pico_viz_robot import PICOVizRobot
from robotreviewer.robots.sample_size_robot import SampleSizeBot

from robotreviewer.data_structures import MultiDict

from robotreviewer import config
import robotreviewer


######
## default annotation pipeline defined here
######
'''
log.info("Loading the robots...")
bots = {"bias_bot": BiasRobot(top_k=3),
        "pico_bot": PICORobot(),
        "pubmed_bot": PubmedRobot(),
        # "ictrp_bot": ICTRPRobot(),
        #"rct_bot": RCTRobot(),
        #"pico_viz_bot": PICOVizRobot(),
        "sample_size_bot":SampleSizeBot()}

log.info("Robots loaded successfully! Ready...")
'''

# lastly wait until Grobid is connected
pdf_reader.connect()

# start up Celery service
app = Celery('ml_worker', backend='amqp://', broker='amqp://')

#####
## connect to and set up database
#####
rr_sql_conn = sqlite3.connect(robotreviewer.get_data('uploaded_pdfs/uploaded_pdfs.sqlite'), detect_types=sqlite3.PARSE_DECLTYPES,  check_same_thread=False)


c = rr_sql_conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS doc_queue (id INTEGER PRIMARY KEY, report_uuid TEXT, pdf_uuid TEXT, pdf_hash TEXT, pdf_filename TEXT, pdf_file BLOB, timestamp TIMESTAMP)')

c.execute('CREATE TABLE IF NOT EXISTS article(id INTEGER PRIMARY KEY, report_uuid TEXT, pdf_uuid TEXT, pdf_hash TEXT, pdf_file BLOB, annotations TEXT, timestamp TIMESTAMP, dont_delete INTEGER)')
c.close()
rr_sql_conn.commit()

@worker_init.connect
def on_worker_init(**_):
    global bots
    log.info("Loading the robots...")
    bots = {"bias_bot": BiasRobot(top_k=3),
            "pico_bot": PICORobot(),
            "pubmed_bot": PubmedRobot(),
            # "ictrp_bot": ICTRPRobot(),
            #"rct_bot": RCTRobot(),
            #"pico_viz_bot": PICOVizRobot(),
            "sample_size_bot":SampleSizeBot()}

    log.info("Robots loaded successfully! Ready...")


@app.task
def annotate(report_uuid):
    """
    takes a report uuid as input
    searches for pdfs using that id,
    then saves annotations in database
    """
    pdf_uuids, pdf_hashes, filenames, blobs, timestamps = [], [], [], [], []

    c = rr_sql_conn.cursor()

    #import pdb; pdb.set_trace()
    
    # load in the PDF data from the queue table
    for pdf_uuid, pdf_hash, filename, pdf_file, timestamp in c.execute("SELECT pdf_uuid, pdf_hash, pdf_filename, pdf_file, timestamp FROM doc_queue WHERE report_uuid=?", (report_uuid, )):
        pdf_uuids.append(pdf_uuid)
        pdf_hashes.append(pdf_hash)
        filenames.append(filename)
        blobs.append(pdf_file)
        timestamps.append(timestamp)

    c.close()

    current_task.update_state(state='PROGRESS', meta={'process_percentage': 25, 'task': 'reading PDFs'})
    articles = pdf_reader.convert_batch(blobs)
    parsed_articles = []


    current_task.update_state(state='PROGRESS', meta={'process_percentage': 50, 'task': 'parsing text'})
    # tokenize full texts here
    for doc in nlp.pipe((d.get('text', u'') for d in articles), batch_size=1, n_threads=config.SPACY_THREADS):
        parsed_articles.append(doc)

    # adjust the tag, parse, and entity values if these are needed later
    for article, parsed_text in zip(articles, parsed_articles):
        article._spacy['parsed_text'] = parsed_text
        current_task.update_state(state='PROGRESS',meta={'process_percentage': 75, 'task': 'doing machine learning'})
    

    for pdf_uuid, pdf_hash, filename, blob, data, timestamp in zip(pdf_uuids, pdf_hashes, filenames, blobs, articles, timestamps):
        # "pico_viz_bot",
        data = annotate_study(data, bot_names=["pubmed_bot", "bias_bot", "pico_bot", "sample_size_bot"])
        #data = annotate_study(data, bot_names=["bias_bot"])
        data.gold['pdf_uuid'] = pdf_uuid
        data.gold['filename'] = filename
        c = rr_sql_conn.cursor()
        c.execute("INSERT INTO article (report_uuid, pdf_uuid, pdf_hash, pdf_file, annotations, timestamp, dont_delete) VALUES(?, ?, ?, ?, ?, ?, ?)", (report_uuid, pdf_uuid, pdf_hash, sqlite3.Binary(blob), data.to_json(), timestamp, config.DONT_DELETE))
        rr_sql_conn.commit()
        c.close()

    # finally delete the PDFs from the queue
    c = rr_sql_conn.cursor()
    c.execute("DELETE FROM doc_queue WHERE report_uuid=?", (report_uuid, ))
    rr_sql_conn.commit()
    c.close()
    current_task.update_state(state='SUCCESS', meta={'process_percentage': 100, 'task': 'done!'})
    return {"process_percentage": 100, "task": "completed"}


def annotate_study(data, bot_names=["bias_bot"]):
    #
    # ANNOTATION TAKES PLACE HERE
    # change the line below if you wish to customise or
    # add a new annotator
    #
    annotations = annotation_pipeline(bot_names, data)
    return annotations

def annotation_pipeline(bot_names, data):
    # makes it here!
    # rdb.set_trace() 
    for bot_name in bot_names:
        log.debug("Sending doc to {} for annotation...".format(bots[bot_name].__class__.__name__))

        data = bots[bot_name].annotate(data)
        log.debug("{} done!".format(bots[bot_name].__class__.__name__))
    return data
