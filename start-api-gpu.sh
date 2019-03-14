#!/bin/bash

MODEL_PATH="$(pwd)/robotreviewer/data"
docker run --runtime=nvidia --name "robotreviewer-gpu" --volume ${MODEL_PATH}:/var/lib/deploy/robotreviewer/data  --env ROBOTREVIEWER_REST_API=true -d --restart="always" -p 127.0.0.1:5051:5000 robotreviewer-gpu
