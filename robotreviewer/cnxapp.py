from robotreviewer.util import rand_id
from celery import Celery
from celery.result import AsyncResult
from datetime import datetime
import robotreviewer
import sqlite3
import json

import connexion

celery_app = Celery('robotreviewer.ml_worker', backend='amqp://', broker='amqp://')
celery_tasks = {"api_annotate": celery_app.signature('robotreviewer.ml_worker.api_annotate')}

rr_sql_conn = sqlite3.connect(robotreviewer.get_data('uploaded_pdfs/uploaded_pdfs.sqlite'), detect_types=sqlite3.PARSE_DECLTYPES,  check_same_thread=False)

def queue_documents(body):
    report_uuid = rand_id()
    c = rr_sql_conn.cursor()
    c.execute("INSERT INTO api_queue (report_uuid, uploaded_data, timestamp) VALUES (?, ?, ?)", (report_uuid, json.dumps(body), datetime.now()))
    rr_sql_conn.commit()
    c.close()
    # send async request to Celery
    celery_tasks['api_annotate'].apply_async((report_uuid, ), task_id=report_uuid)
    return json.dumps({"report_id": report_uuid})


def report_status(report_id):
    '''
    check and return status of celery annotation process
    '''
    result = AsyncResult(report_id, app=celery_app)
    return json.dumps({"status": result.status})


def report(report_id):
    c = rr_sql_conn.cursor()
    c.execute("SELECT annotations FROM api_done WHERE report_uuid = ?", (report_id, ))
    result = c.fetchone()    
    c.close()
    return result[0]

import connexion
app = connexion.FlaskApp(__name__, specification_dir='api/', port=5000, server='gevent')
app.add_api('robotreviewer_api.yml') 
