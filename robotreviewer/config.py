"""
RobotReviewer configuration file
"""

# grobid
USE_GROBID = True
GROBID_PATH = "/Users/byron/dev/grobid" # "/Users/iain/Code/grobid-grobid-parent-0.4.0"
GROBID_HOST = "http://localhost:8080/" # must have http:// to start with
TEMP_PDF = "/Users/byron/dev/robotreviewer3/robotreviewer/textprocessing/pdftmp"
TEMP_ZIP = "/Users/byron/dev/robotreviewer3/robotreviewer/textprocessing/ziptmp"

GROBID_THREADS = 8 # max number of articles to send to Grobid at once

# spacy
SPACY_THREADS = 8









