"""
RobotReviewer configuration file
"""

import os
import json

CONFIG_KEY = "ROBOTREVIEWER";

def merge_dicts(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def json_config():
    abs_dir = os.path.dirname(__file__) # absolute path to here
    path = os.path.join(abs_dir, "config.json")
    with open(path) as p:
        return json.load(p)[CONFIG_KEY.lower()]

def val(v):
    truthy = ("yes", "true", "t", "1")
    falsey = ("no", "false", "f", "0")
    v_lower = v.lower()
    if(v_lower in truthy or v_lower in falsey):
        return v_lower in truthy
    elif(v.isdigit()):
        return int(v)
    else:
        return v

def environ_config():
    env = dict(os.environ)
    kf = lambda k: k.replace(CONFIG_KEY + "_", "").lower()

    return { kf(k) : val(env[k]) for k in env.keys() if k.startswith(CONFIG_KEY) }

def config():
    return merge_dicts(json_config(), environ_config())

def export_config(cfg):
    for k, v in cfg.items():
        globals()[k.upper()] = v

export_config(config())
