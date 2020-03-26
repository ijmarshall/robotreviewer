#!/bin/bash

MODEL_PATH="$(pwd)/robotreviewer/data"
docker run --name "robotreviewer-beta" --volume ${MODEL_PATH}:/var/lib/deploy/robotreviewer/data  --env ROBOTREVIEWER_REST_API=true -d --restart="always" -p 127.0.0.1:5055:5000 robotreviewer-beta
