import os
import base64

def rand_id():
    return base64.urlsafe_b64encode(os.urandom(16))[:21].decode('utf-8')
