import logging
from datetime import datetime, timedelta
import robotreviewer
import sqlite3

LOCAL_PATH = "robotreviewer/uploads"

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s %(asctime)s: %(message)s')
log = logging.getLogger(__name__)


def cleanup_database(days=1):
    """
    remove any PDFs which have been here for more than
    1 day, then compact the database
    """
    log.info('Cleaning up database with uploaded_pdfs')
    conn = sqlite3.connect(robotreviewer.get_data('uploaded_pdfs/uploaded_pdfs.sqlite'),
                           detect_types=sqlite3.PARSE_DECLTYPES)

    d = datetime.now() - timedelta(days=days)
    c = conn.cursor()
    c.execute("DELETE FROM article WHERE timestamp < datetime(?) AND dont_delete=0", [d])
    conn.commit()
    # conn.execute("VACUUM")  # make the database smaller again
    # conn.commit()
    conn.close()


if __name__ == "__main__":
    log.info("Running clean-up task")
    cleanup_database(days=1)
    log.info("Clean-up task complete")
