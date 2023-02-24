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


if __name__ == "__main__":
    print("Running Naive Bayes tests:")
    unittest.main()
