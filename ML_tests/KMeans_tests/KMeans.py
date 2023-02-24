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

    # def test_perfectnegativeslope(self):
    #     W = linear_regression_normal_equation(self.X2, self.y2)
    #     boolean_array = np.isclose(W, self.W2_correct)
    #     self.assertTrue(boolean_array.all())

    # def test_multipledimension(self):
    #     W = linear_regression_normal_equation(self.X3, self.y3)
    #     print(W)
    #     print(self.W3_correct)
    #     boolean_array = np.isclose(W, self.W3_correct)
    #     self.assertTrue(boolean_array.all())

    # def test_zeros(self):
    #     W = linear_regression_normal_equation(self.X4, self.y4)
    #     boolean_array = np.isclose(W, self.W4_correct)
    #     self.assertTrue(boolean_array.all())

    # def test_noisydata(self):
    #     W = linear_regression_normal_equation(self.X5, self.y5)
    #     boolean_array = np.isclose(W, self.W5_correct, atol=1e-3)
    #     self.assertTrue(boolean_array.all())


if __name__ == "__main__":
    print("Running KMeans tests:")
    unittest.main()
