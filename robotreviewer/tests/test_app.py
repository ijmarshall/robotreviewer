
import unittest

from robotreviewer.app import get_study_name
from robotreviewer.data_structures import MultiDict

class TestApp(unittest.TestCase):

    def test_get_study_name(self):
        test_json = '{"gold": {                                                  \
            "authors": [                                                         \
                {"forename": "Carlos", "lastname": "Noronha", "initials": "C"},  \
                {"forename": "Neto C", "lastname": "", "initials": "NC"},        \
                {"forename": "Sabina S B", "lastname": "Maia", "initials": "SS"} \
            ]}}'
        mdict = MultiDict()
        mdict.load_json(test_json)
        study_name = "Noronha et al."
        self.assertEqual(get_study_name(mdict), study_name)
        test_json = '{"gold": {                                                  \
            "authors": [                                                         \
                {"forename": "Carlos", "lastname": "Noronha", "initials": "C"},  \
                {"forename": "Neto C", "lastname": "", "initials": "NC"}         \
            ]}}'
        mdict.load_json(test_json)
        study_name = "Noronha et al."
        self.assertEqual(get_study_name(mdict), study_name)
        test_json = '{"gold": {                                                \
            "authors": [                                                       \
                {"forename": "Carlos", "lastname": "Noronha", "initials": "C"} \
            ]}}'
        mdict.load_json(test_json)
        study_name = "Noronha, Carlos C."
        self.assertEqual(get_study_name(mdict), study_name)
