import unittest
import pandas as pd
import numpy as np
from ModelBootStrapper import ModelBootStrapper
from lightgbm import LGBMClassifier as clf


class ModelBootStrapperMetricsTestCase(unittest.TestCase):
    def test_calculate_ppv(self):
        m = ModelBootStrapper(estimator=clf(), n_boot=5)
        X = pd.DataFrame(np.random.normal(size=(50, 2)))
        y = pd.Series(np.random.binomial(n=1, p=0.5, size=50))
        m.fit(X, y)

        ppv_score = m.calculate_ppv(X, y)
        assert isinstance(
            ppv_score, tuple
        ), "Returned PPV score should be of type tuple"
        assert isinstance(
            ppv_score[0], float
        ), "Returned PPV score first value should be of type float"
        assert isinstance(
            ppv_score[1], tuple
        ), "Returned PPV score second value should be of type float"
        assert isinstance(
            ppv_score[1][0], tuple
        ), "Returned PPV score second value should contain two floats"
        assert isinstance(
            ppv_score[1][1], tuple
        ), "Returned PPV score second value should contain two floats"

        with self.assertRaises(TypeError) as cm:
            m.calculate_ppv(X, y, threshold=2)
        self.assertEqual(str(cm.exception), "threshold has to be of type float")

        with self.assertRaises(ValueError) as cm:
            m.calculate_ppv(X, y, threshold=2.1)
        self.assertEqual(str(cm.exception), "threhols has to be between [0, 1]")

        with self.assertRaises(ValueError) as cm:
            m.calculate_ppv(X, np.zeros_like(y) + 2)
        self.assertEqual(str(cm.exception), "y_true must include only [0, 1]")


if __name__ == "__main__":
    unittest.main()
