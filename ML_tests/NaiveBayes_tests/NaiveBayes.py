# Import folder where sorting algorithms
import sys
import unittest
import numpy as np

# For importing from different folders
# OBS: This is supposed to be done with automated testing,
# hence relative to folder we want to import from
sys.path.append("ML/algorithms/naivebayes")

# If run from local:
# sys.path.append('../../ML/algorithms/linearregression/')
from naivebayes import NaiveBayes


class NaiveBayes(unittest.TestCase):
    def setUp(self):
        # test cases we want to run
        self.num_examples = 3
        self.num_features = 5
        self.num_classes = 3
        self.eps = 1e-6

    def test_correctShape(self):

        instance = NaiveBayes(np.array(

            [
                [1,1,1,4,6],
                [2,3,3,3,6],
                [4,6,7,4,7]
            ]

        ), [1,1,1,3,5,6,7,3,5,7,6])

        self.assertTrue((instance.num_examples, instance.num_classes == (self.num_examples, self.num_features))

    def test_correctNumFeatures(self):

        instance = NaiveBayes(np.array(

            [
                [1,1,1,4,6],
                [2,3,3,3,6],
                [4,6,7,4,7]
            ]

        ), [1,1,1,3,5,6,7,3,5,7,6])
        self.assertTrue(instance.num_features == self.num_features)

    def test_correctNumExamples(self):

        instance = NaiveBayes(np.array(

            [
                [1,1,1,4,6],
                [2,3,3,3,6],
                [4,6,7,4,7]
            ]

        ), [1,1,1,3,5,6,7,3,5,7,6])
        self.assertTrue(instance.num_examples == self.num_examples)

    def test_correctEps(self):

        instance = NaiveBayes(np.array(

            [
                [1,1,1,4,6],
                [2,3,3,3,6],
                [4,6,7,4,7]
            ]

        ), [1,1,1,3,5,6,7,3,5,7,6])
        self.assertTrue(instance.eps == self.eps)

if __name__ == "__main__":
    print("Running Naive Bayes tests:")
    unittest.main()
