FROM python:3.6-slim

WORKDIR /bert

COPY requirements.txt .

RUN pip install -r requirements.txt

CMD bert-serving-start -model_dir $SCIBERT_PATH_MODEL -num_worker=1
