#!/usr/bin/env python
from robotreviewer.app import app
from gevent.wsgi import WSGIServer

http_server = WSGIServer(('', 5000), app)
http_server.serve_forever()
