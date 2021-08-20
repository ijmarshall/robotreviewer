import logging
import os
from datetime import datetime
import json

from celery.result import AsyncResult
import connexion
from connexion.exceptions import OAuthProblem
import robotreviewer
from robotreviewer.app import create_celery_app, str2bool
from robotreviewer.util import rand_id
import sqlite3


DEBUG_MODE = str2bool(os.environ.get("DEBUG", "true"))
LOG_LEVEL = (logging.DEBUG if DEBUG_MODE else logging.INFO)
logging.basicConfig(level=LOG_LEVEL, format='[%(levelname)s] %(name)s %(asctime)s: %(message)s')
log = logging.getLogger(__name__)

rr_sql_conn = sqlite3.connect(robotreviewer.get_data('uploaded_pdfs/uploaded_pdfs.sqlite'), detect_types=sqlite3.PARSE_DECLTYPES,  check_same_thread=False)

celery_app, celery_tasks = create_celery_app()

def auth(api_key, required_scopes):
    info = robotreviewer.config.API_KEYS.get(api_key, None)
    if not info:
        raise OAuthProblem('Invalid token')
    return info


def queue_documents(body):
    report_uuid = rand_id()
    c = rr_sql_conn.cursor()
    c.execute("INSERT INTO api_queue (report_uuid, uploaded_data, timestamp) VALUES (?, ?, ?)", (report_uuid, json.dumps(body), datetime.now()))
    rr_sql_conn.commit()
    c.close()
    # send async request to Celery
    log.debug(f'Running celery task: api_annotate for report: {report_uuid}')
    celery_tasks['api_annotate'].apply_async((report_uuid, ), task_id=report_uuid)
    return {"report_id": report_uuid}


def report_status(report_id):
    '''
    check and return status of celery annotation process
    '''
    log.debug(f"Calling AsyncResult to annotate status of task: {report_id}")
    result = AsyncResult(report_id, app=celery_app)
    log.debug(f"result: {result}")
    return {"state": result.state, "meta": result.result}


def report(report_id):
    c = rr_sql_conn.cursor()
    c.execute("SELECT annotations FROM api_done WHERE report_uuid = ?", (report_id, ))
    result = c.fetchone()
    c.close()
    return json.loads(result[0])


def create_app():
    app = connexion.FlaskApp(__name__, specification_dir='api/', port=5000, server='gevent')
    app.add_api('robotreviewer_api.yml')
    log.info("Welcome to RobotReviewer API server :)")
    return app


app = create_app()
