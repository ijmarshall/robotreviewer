#!/bin/sh

OPTS=$CMD

if [ -z "$OPTS" ]
then
    OPTS=$1
fi

case "$OPTS" in

celery)
    echo "[entrypoint.sh] Starting Celery"
    cd /var/lib/deploy/ && celery -A robotreviewer.ml_worker worker --loglevel=info --concurrency=1 --pool=solo
    ;;
web)
    echo "[entrypoint.sh] Starting RobotReviewer Web server"
    cd /var/lib/deploy/ && gunicorn --worker-class gevent --workers $GUNICORN_WORKERS --timeout $GUNICORN_WORKER_TIMEOUT -b 0.0.0.0:5000 server:app
    ;;
web-dev)
    echo "[entrypoint.sh] Starting RobotReviewer Web server in development mode"
    export LC_ALL=C.UTF-8
    export LANG=C.UTF-8
    export FLASK_APP=server:app
    export FLASK_ENV=development
    cd /var/lib/deploy/ && flask run --host 0.0.0.0 --port 5000 --eager-loading --no-reload
    ;;
api)
    echo "[entrypoint.sh] Starting RobotReviewer API server"
    cd /var/lib/deploy/ && gunicorn --worker-class gevent --workers $GUNICORN_WORKERS --timeout $GUNICORN_WORKER_TIMEOUT -b 0.0.0.0:5001 server_api:app
    ;;
api-dev)
    echo "[entrypoint.sh] Starting RobotReviewer API server in development mode"
    export LC_ALL=C.UTF-8
    export LANG=C.UTF-8
    export FLASK_APP=server_api:flask_app
    export FLASK_ENV=development
    cd /var/lib/deploy/ && flask run --host 0.0.0.0 --port 5001 --eager-loading --no-reload
    ;;
test)
    echo "[entrypoint.sh] Running RobotReviewer unit tests"
    cd /var/lib/deploy/ && python -m unittest &
    sleep infinity
    ;;
*)
    if [ ! -z "$(which $1)" ]
    then
        $@
    else
        echo "Invalid command"
        exit 1
    fi
    ;;
esac
