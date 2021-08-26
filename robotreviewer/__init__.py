"""
RobotReviewer
"""

# Authors:  Iain Marshall <mail@ijmarshall.com>
#           Joel Kuiper <me@joelkuiper.com>
#           Byron Wallce <byron.wallace@utexas.edu>

import os

_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_ROOT = os.path.join(_ROOT, 'data') 

def get_data(path):
    return os.path.join(DATA_ROOT, path)
