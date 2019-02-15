#!/bin/bash

echo "building image"
docker build --build-arg OSVER="nvidia/cuda:9.0-cudnn7-runtime" --build-arg TFVER="tensorflow-gpu" -t robotreviewer-gpu .
