ARG OSVER=tensorflow/tensorflow:1.12.0-py3
FROM $OSVER

WORKDIR /bert

RUN apt-get update
RUN pip install bert-serving-server

CMD bert-serving-start -model_dir $SCIBERT_PATH_MODEL -num_worker=1

