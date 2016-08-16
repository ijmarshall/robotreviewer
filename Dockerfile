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

# Get the source
ADD deploy.tar.gz /var/lib/deploy/src

RUN chown -R deploy.deploy /var/lib/deploy

## From here on we're the deploy user
USER deploy

# get grobid
RUN mkdir /var/lib/deploy/tmp
RUN cd /var/lib/deploy/tmp && wget https://github.com/kermitt2/grobid/archive/grobid-parent-0.4.0.zip
RUN cd /var/lib/deploy/tmp && unzip grobid-parent-0.4.0.zip && mv grobid-grobid-parent-0.4.0 grobid
RUN cd /var/lib/deploy/tmp/grobid && mvn -Dmaven.test.skip=true clean install
RUN cd /var/lib/deploy/tmp/ && mv grobid /var/lib/deploy/grobid && rm -rf /var/lib/deploy/tmp

# install Anaconda
RUN aria2c -s 16 -x 16 -k 30M https://repo.continuum.io/archive/Anaconda2-4.1.1-Linux-x86_64.sh -o /var/lib/deploy/Anaconda.sh
RUN cd /var/lib/deploy && bash Anaconda.sh -b && rm -rf Anaconda.sh
ENV PATH=/var/lib/deploy/anaconda2/bin:$PATH

RUN conda config --add channels spacy
RUN conda install flask numpy scipy scikit-learn spacy
RUN python -m spacy.en.download

# install Python dependencies
ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# compile client side assets
RUN cd /var/lib/deploy/src/robotreviewer/ &&  r.js -o static/build.js && rm -rf static && mv build static

EXPOSE 5000
USER deploy
ENV HOME /var/lib/deploy
ENV ROBOTREVIEWER_GROBID_PATH=/var/lib/deploy/grobid
ENV ROBOTREVIEWER_GROBID_HOST=http://0.0.0.0:8080
ENV DEV false
ENTRYPOINT ["/var/lib/deploy/src/server"]