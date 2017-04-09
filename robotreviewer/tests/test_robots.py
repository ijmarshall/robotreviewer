import json
import unittest
import os


from robotreviewer.robots.rationale_robot import BiasRobot
from robotreviewer.data_structures import MultiDict

class TestBiasRobot(unittest.TestCase):

    br = BiasRobot()
    examples_path = os.path.dirname(__file__) + "/examples/"

    def test_simple_borda_count(self):
        ''' tests for BiasRobot.simple_borda_count(a, b, weights=None) '''
        a = [2, 1, 0, 3, 4]
        b = [2, 1, 4, 0, 3]
        test_output = [3, 4, 0, 1, 2]
        output = self.br.simple_borda_count(a, b)
        self.assertEqual(output, test_output)
        a = [9, 2, 1, 0, 3, 4]
        output = self.br.simple_borda_count(a, b)
        self.assertEqual(output, test_output)
        output = self.br.simple_borda_count(a, b, weights=[2.0, 2.0])
        self.assertEqual(output, test_output)
        output = self.br.simple_borda_count(a, b, weights=[1.0, 2.0])
        test_output = [3, 0, 4, 1, 2]
        self.assertEqual(output, test_output)
        a = [90, 15, 65, 45, 78, 71, 69, 94, 57, 18, 92, 60, 14, 79, 70, 59, 7,
             85, 54, 89, 86, 66, 72, 80, 9, 93, 63, 46, 30, 8, 44, 3, 58,
             13, 67, 42, 84, 29, 40, 6, 77, 11, 61, 39, 81, 34, 88, 38, 28,
             43, 62, 47, 22, 5, 97, 95, 96, 32, 64, 91, 20, 53, 27, 21, 73,
             33, 16, 49, 24, 19, 68, 83, 17, 2, 41, 55, 51, 56, 23, 82, 31,
             87, 25, 37, 50, 35, 36, 76, 75, 10, 74, 12, 4, 1, 0, 52, 26, 48]
        b = [44, 34, 53, 38, 86, 18, 96, 97, 95, 63, 94, 32, 24, 70, 93, 31,
             85, 90, 69, 15, 50, 9, 68, 21, 67, 62, 76, 23, 42, 78, 87, 5, 71,
             22, 30, 4, 45, 79, 58, 89, 39, 88, 12, 80, 3, 6, 84, 83, 81, 74,
             60, 57, 73, 16, 13, 77, 28, 82, 20, 33, 25, 64, 40, 75, 91, 36,
             72, 19, 61, 11, 26, 17, 92, 66, 59, 52, 7, 56, 37, 47, 41, 49, 54,
             65, 0, 35, 55, 8, 43, 27, 51, 2, 10, 48, 14, 46, 1, 29]
        test_output = [48, 1, 10, 0, 35, 52, 26, 51, 2, 37, 55, 41, 56, 27, 36,
                       75, 49, 17, 25, 74, 43, 19, 82, 29, 12, 47, 4, 33, 91,
                       46, 16, 64, 20, 83, 8, 73, 76, 87, 11, 61, 14, 23, 28,
                       50, 40, 54, 31, 77, 66, 7, 68, 81, 59, 72, 13, 88, 21,
                       22, 65, 5, 6, 39, 84, 92, 24, 3, 62, 58, 32, 80, 42, 53,
                       95, 30, 96, 60, 97, 57, 67, 89, 38, 79, 34, 9, 45, 93,
                       71, 63, 78, 85, 44, 70, 69, 86, 15, 90, 94, 18]
        output = self.br.simple_borda_count(a, b)
        self.assertEqual(output, test_output)

    def dont_test_annotate(self):
        with open(self.examples_path + "/data_input.json") as input_file:
            inp = input_file.read()
        data = MultiDict()
        data.load_json(inp)
        data = self.br.annotate(data)

        with open(self.examples_path + "/data_output.json") as output_file:
            out = output_file.read()
        post_data = MultiDict()
        post_data.load_json(out)

        pre = json.loads(data.to_json())
        post = json.loads(out)

        for key in pre:
            print(key)
            self.assertEqual(key in post, True)
        self.assertEqual(len(pre), len(post))

    def dont_test_get_domains(self):
        d1 = self.br.bias_domains
        d2 = self.br.get_domains()
        self.assertEqual(len(d1), len(d2))
        for d in d1:
            self.assertEqual(d1[d] in d2, True)
