ARG OSVER=ubuntu:16.04
FROM $OSVER

ENV DEBIAN_FRONTEND noninteractive

# create deploy user
RUN useradd --create-home --home /var/lib/deploy deploy

# install apt-get requirements
ADD apt-requirements.txt /tmp/apt-requirements.txt
RUN apt-get -qq update -y
RUN xargs -a /tmp/apt-requirements.txt apt-get install -y --no-install-recommends && apt-get clean

# Certs
RUN mkdir -p /etc/pki/tls/certs && \
    ln -s /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt

# node.js and utils
RUN add-apt-repository ppa:chris-lea/node.js
RUN apt-get install -y nodejs npm && npm update
ENV NODE_PATH $NODE_PATH:/usr/local/lib/node_modules
RUN npm install -g requirejs
RUN ln -s /usr/bin/nodejs /usr/bin/node

# Gradle requires jdk to work! Java incoming!
RUN apt-get -y install openjdk-8-jdk wget unzip
RUN java -version # check that java works

# install gradle the annoying thing
RUN mkdir /opt/gradle
RUN curl -L https://services.gradle.org/distributions/gradle-4.10.2-bin.zip -o gradle-4.10.2.zip && \
    unzip -d /opt/gradle gradle-4.10.2.zip && \
    rm gradle-4.10.2.zip
#ENV GRADLE_HOME=/usr/local/gradle-4.10.2
ENV PATH=/opt/gradle/gradle-4.10.2/bin:$PATH
#ENV PATH=$PATH:$GRADLE_HOME/bin
RUN gradle -v # check that gradle works

RUN cd /var/lib/deploy/ && wget https://github.com/kermitt2/grobid/archive/0.5.1.zip -O grobid.zip && \
    unzip grobid.zip && \
    cd /var/lib/deploy/grobid-0.5.1 && \
    gradle clean install && \
#    mvn -Dmaven.test.skip=true clean install && \
    rm -f /var/lib/deploy/grobid.zip

RUN chown -R deploy.deploy /var/lib/deploy/

USER deploy
# install Anaconda
RUN aria2c -s 16 -x 16 -k 30M https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /var/lib/deploy/Anaconda.sh
RUN cd /var/lib/deploy && bash Anaconda.sh -b && rm -rf Anaconda.sh
ENV PATH=/var/lib/deploy/miniconda3/bin:$PATH
ADD robotreviewer_env.yml tmp/robotreviewer_env.yml
RUN conda env create -f tmp/robotreviewer_env.yml
# from https://stackoverflow.com/questions/37945759/condas-source-activate-virtualenv-does-not-work-within-dockerfile
ENV PATH /var/lib/deploy/miniconda3/envs/robotreviewer/bin:$PATH
RUN python -m nltk.downloader punkt stopwords
#RUN python -m spacy.en.download all
RUN python -m spacy download en

ARG TFVER=tensorflow
RUN pip install $TFVER==1.12.0


#strange Theano problem
#ENV MKL_THREADING_LAYER=GNU

# Get data
USER root

RUN mkdir -p /var/lib/deploy/robotreviewer/data
ADD server.py /var/lib/deploy/
ADD run /var/lib/deploy/
ADD robotreviewer /var/lib/deploy/robotreviewer
RUN chown -R deploy.deploy /var/lib/deploy/robotreviewer

USER deploy
VOLUME /var/lib/deploy/src/robotreviewer/data
# compile client side assets
RUN cd /var/lib/deploy/robotreviewer/ && \
    r.js -o static/build.js && \
    mv static static.bak && \
    mv build static && \
    rm -rf static.bak

EXPOSE 5000
ENV HOME /var/lib/deploy

USER root

ENTRYPOINT ["/var/lib/deploy/run"]
