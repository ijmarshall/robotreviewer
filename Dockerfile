FROM ubuntu:16.04
ENV DEBIAN_FRONTEND noninteractive

RUN echo "UTC" > /etc/timezone
RUN dpkg-reconfigure tzdata

# Set locale
RUN locale-gen en_US.UTF-8
RUN update-locale LANG=en_US.UTF-8

ENV LANG C.UTF-8

# create deploy user
RUN useradd --create-home --home /var/lib/deploy deploy

# install apt-get requirements
ADD apt-requirements.txt /tmp/apt-requirements.txt
RUN apt-get update -y
RUN xargs -a /tmp/apt-requirements.txt apt-get install -y

# Certs
RUN mkdir -p /etc/pki/tls/certs
RUN ln -s /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt

# node.js and utils
RUN add-apt-repository ppa:chris-lea/node.js
RUN apt-get install -y nodejs npm && npm update
ENV NODE_PATH $NODE_PATH:/usr/local/lib/node_modules
RUN npm install -g requirejs
RUN ln -s /usr/bin/nodejs /usr/bin/node


RUN chown -R deploy.deploy /var/lib/deploy/

## From here on we're the deploy user
USER deploy
RUN cd /var/lib/deploy/ && wget https://github.com/kermitt2/grobid/archive/grobid-parent-0.4.1.zip -O grobid.zip
RUN cd /var/lib/deploy/ && unzip grobid.zip
RUN cd /var/lib/deploy/grobid-grobid-parent-0.4.1 && mvn -Dmaven.test.skip=true clean install

# install Anaconda
RUN aria2c -s 16 -x 16 -k 30M https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /var/lib/deploy/Anaconda.sh
RUN cd /var/lib/deploy && bash Anaconda.sh -b && rm -rf Anaconda.sh
ENV PATH=/var/lib/deploy/miniconda3/bin:$PATH
run conda install python=3.5.0
run conda config --add channels spacy
run conda install system flask numpy scipy scikit-learn spacy flask-wtf requests pandas gensim mkl mkl-service matplotlib seaborn h5py
run conda install pyqt=4.11
run python -m spacy.en.download

# install Python dependencies
ADD requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt
RUN python -m nltk.downloader punkt stopwords

# Get data
USER root
RUN mkdir -p /var/lib/deploy/robotreviewer/data
ADD server /var/lib/deploy/
ADD run /var/lib/deploy/
ADD robotreviewer /var/lib/deploy/robotreviewer
RUN chown -R deploy:deploy /var/lib/deploy/robotreviewer
USER deploy
VOLUME /var/lib/deploy/src/robotreviewer/data

# compile client side assets
RUN cd /var/lib/deploy/robotreviewer/ &&  r.js -o static/build.js && mv static static.bak && mv build static

EXPOSE 5000
USER deploy
ENV HOME /var/lib/deploy
ENV ROBOTREVIEWER_GROBID_PATH=/var/lib/deploy/grobid-grobid-parent-0.4.1
ENV ROBOTREVIEWER_GROBID_HOST=http://0.0.0.0:8080
ENV DEV false
ENV DEBUG false
ENV KERAS_BACKEND=theano
ENTRYPOINT ["/var/lib/deploy/run"]
