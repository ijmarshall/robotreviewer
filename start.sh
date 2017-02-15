#!/bin/bash


MODEL_PATH="$(pwd)/robotreviewer"
docker run --name "robotreviewer" --volume ${MODEL_PATH}:/var/lib/deploy/src -d robotreviewer
