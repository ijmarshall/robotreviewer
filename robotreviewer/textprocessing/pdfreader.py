"""
Simple interface to Grobid

Requires Grobid to be running, by default on localhost:8080
This can be set in config.py
"""

from robotreviewer import config
from robotreviewer.data_structures import MultiDict
import requests
import xml.etree.cElementTree as ET

try:
    from cStringIO import StringIO # py2
except ImportError:
    from io import StringIO # py3

try:
    import urlparse
except ImportError:
    from urllib import parse as urlparse

import subprocess
import os
import logging
import time
import atexit
import codecs
import json
# Author:  Iain Marshall <mail@ijmarshall.com>

log = logging.getLogger(__name__)

class Grobid():
    """
    starts up Grobid as a service on default port (should be 8080)
    checks to see if it's working before returning the open process
    """
    def __init__(self, check_delay=2):

        self.devnull = open(os.devnull, 'wb')
        atexit.register(self.cleanup)
        grobid_process = subprocess.Popen(['mvn', '-q', '-Dmaven.test.skip=true', 'jetty:run-war'], cwd=os.path.join(config.GROBID_PATH, 'grobid-service'), stdout=self.devnull, stderr=subprocess.STDOUT) # skip tests since they will not run properly from python subprocess

        connected = False

        log.info('Waiting for Grobid to start up...')
        while connected == False:
            try:
                r = requests.get('http://localhost:8080')
                r.raise_for_status() # raise error if not HTTP: 200
                connected = True
            except:
                time.sleep(check_delay)

        log.info('Grobid connection success :)')
        self.connection = grobid_process # not explicitly needed... but maintains process open

    def cleanup(self):
        self.connection.kill()
        self.devnull.close()



class PdfReader():

    def __init__(self):
        self.url = urlparse.urljoin(config.GROBID_HOST, 'processFulltextDocument')        
        log.info('Attempting to start Grobid sever...')
        self.grobid_process = Grobid()
        log.info('Success! :)')

    def cleanup(self):
        self.grobid_process.cleanup() 

    def convert(self, pdf_binary):
        """
        returns MultiDict containing document information
        """
        out = self.parse_xml(self.run_grobid(pdf_binary))
        return out

    def run_grobid(self, pdf_binary, MAX_TRIES=5):
        files = {'input': StringIO(pdf_binary)}
        r = requests.post(self.url, files=files)
        
        try:
            r.raise_for_status() # raise error if not HTTP: 200
        except Exception:
            log.info("oh dear... post request to grobid failed for %s. exception below." % filename)
            log.info(r.text)
            raise

        # TEMPOARY (MAYBE) TODO - remove
        # save output for debugging
        # tmp_filename = os.path.join(config.TEMP_PDF, 'tmp.xml')
        # with codecs.open(tmp_filename, 'w', 'utf-8') as f:
        #     f.write(r.text)        
        # return r.text

    def parse_xml(self, xml_string):
        output = MultiDict()
        full_text_bits = []
        author_list = []
        author_bits = []
        path = []
        for event, elem in ET.iterparse(StringIO(xml_string.encode('utf-8')),events=("start", "end")):
            if event == 'start':
                path.append(elem.tag)
            elif event == 'end':
                if elem.tag=='{http://www.tei-c.org/ns/1.0}abstract':
                    output.grobid['abstract'] = (self._extract_text(elem))
                elif elem.tag=='{http://www.tei-c.org/ns/1.0}title' and '{http://www.tei-c.org/ns/1.0}titleStmt' in path:
                    output.grobid['title'] = self._extract_text(elem)
                elif elem.tag in ['{http://www.tei-c.org/ns/1.0}head', '{http://www.tei-c.org/ns/1.0}p']:
                    full_text_bits.extend([self._extract_text(elem), '\n'])
                elif elem.tag=='{http://www.tei-c.org/ns/1.0}forename':
                    author_bits.append(self._extract_text(elem))
                elif elem.tag=='{http://www.tei-c.org/ns/1.0}surname':
                    author_bits.append(self._extract_text(elem))
                    author_list.append(author_bits)
                    author_bits = []

                path.pop()
        output.grobid['text'] = '\n'.join(full_text_bits)
        output.grobid['authors'] = author_bits
        return output

    def _extract_text(self, elem):
        return ''.join([s.decode("utf-8") for s in ET.tostringlist(elem, method="text", encoding="utf-8") if s is not None]).strip() # don't ask...

def main():
    pass


if __name__ == '__main__':
    main()