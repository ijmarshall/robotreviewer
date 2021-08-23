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
    python /var/lib/deploy/server.py
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
