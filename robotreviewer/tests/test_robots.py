import unittest

from robotreviewer.robots.rationale_robot import BiasRobot

class TestBiasRobot(unittest.TestCase):

    br = BiasRobot()

    def test_annotate(self):
        pass

    def test_marginalia(self):
        pass

    def test_get_domains(self):
        d1 = self.br.bias_domains
        d2 = self.br.get_domains()
        self.assertEqual(len(d1), len(d2))
        for d in d1:
            self.assertEqual(d1[d] in d2, True)
