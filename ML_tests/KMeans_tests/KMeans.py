# Import folder where sorting algorithms
import sys
import unittest
import numpy as np

# For importing from different folders
# OBS: This is supposed to be done with automated testing,
# hence relative to folder we want to import from
sys.path.append("ML/algorithms/kmeans")

# If run from local:
sys.path.append('../../ML/algorithms/linearregression/')
from kmeansclustering import KMeansClustering


class TestKMeansClustering(unittest.TestCase):
    def setUp(self):
        # test cases we want to run
        self.K = 10
        self.max_iterations = 100
        self.plot_figure = True
        self.num_examples = 3
        self.num_features = 3
        

    def test_correctClustersCount(self):

        instance = KMeansClustering(np.array(

            [
                [1,1,1],
                [2,3,3],
                [4,6,7]
            ]

        ), self.K)
        print(instance.K)
        print(instance.max_iterations)
        self.assertTrue(instance.K == self.K)

    def test_correctNumFeatures(self):

        instance = KMeansClustering(np.array(

            [
                [1,1,1],
                [2,3,3],
                [4,6,7]
            ]

        ), self.K)
        self.assertTrue(instance.num_features == self.num_features)

    def test_correctNumExamples(self):

        instance = KMeansClustering(np.array(

            [
                [1,1,1],
                [2,3,3],
                [4,6,7]
            ]

        ), self.K)
        self.assertTrue(instance.num_examples == self.num_examples)

    def test_correctNumIterations(self):

        instance = KMeansClustering(np.array(

            [
                [1,1,1],
                [2,3,3],
                [4,6,7]
            ]

        ), self.K)
        self.assertTrue(instance.max_iterations == self.max_iterations)

if __name__ == "__main__":
    print("Running KMeans tests:")
    unittest.main()
