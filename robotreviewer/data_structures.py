"""
RobotReviewer data structures
"""

import json

class MultiDict():
    """
    Handles storing data from multiple annotators of varying quality
    Currently returns the best quality according the order of annotators
    in a simple list.
    Can be extended in future for more sophisticated orders
    """

    def __init__(self, authorities=None, default_authority=None):

        if authorities:
            self.authorities = authorities
        else:
            # list of authorities from most to least trusted
            self.authorities = ["gold", "human", "mendeley", "pubmed", "ml", "grobid", "dubious", "_spacy"]

        if (default_authority is not None) and (default_authority not in self.authorities):
            raise ValueError('default value is not present in authority list')

        self.default_authority = default_authority

        self.data = {k:{} for k in self.authorities}

    def __getitem__(self, key):
        """
        return value from best authority which exists
        """
        for authority in self.authorities:
            rslt = self.data[authority].get(key)
            if rslt:
                return rslt
        else:
            raise KeyError(u'Key "{}" not present in any of the authorities of this MultiDict'.format(key))

    def get(self, key, default=None):
        """
        return value from best authority which exists
        """
        for authority in self.authorities:
            rslt = self.data[authority].get(key)
            if rslt:
                return rslt
        else:
            return default

    def get_authority(self, key):
        """
        return value from best authority which exists
        in tuple alongside authority name
        """
        for authority in self.authorities:
            rslt = self.data[authority].get(key)
            if rslt:
                return (authority, rslt)
        else:
            return None

    def __repr__(self):
        return self.data.__repr__()

    def __getattr__(self, authority):
        """
        to allow accessing the authorities as an attribute
        """
        return self.data[authority]


    def items(self):
        """
        gets keys across *all* authorities, and returns the
        highest authority each time
        """
        out = {}
        for authority in reversed(self.authorities):
            out.update(self.data[authority])
        return out

    def keys(self):
        """
        gets keys across *all* authorities
        """
        return self.items().keys()

    def values(self):
        """
        gets values across *all* authorities
        """
        return self.items().values()

    def to_json(self):
        """
        saves all the annotations to json
        """
        return json.dumps({k: (v if not k.startswith('_') else {}) for k, v in self.data.items()})

    def load_json(self, json_str):
        """
        loads data from json string
        doesn't do any checking...
        """
        self.data = json.loads(json_str)

    def visible_data(self):
        """
        return the stuff which is supposed to be visible
        """
        return {k: v for k, v in self.data.items() if not k.startswith('_')}

    @classmethod
    def from_dict(this_class, dct):
        out = this_class()
        for k, v in dct.items():
            out[k] = v
        return out

    # doesn't really iterate, but for compatibility
    iteritems = items
    iterkeys = keys
    itervalues = values
