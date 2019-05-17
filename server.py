#!/usr/bin/env python
from robotreviewer import config

from gevent.pywsgi import WSGIServer

if config.REST_API==False:
    from robotreviewer.app import app
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
else:
    from robotreviewer import cnxapp
    cnxapp.app.run()
