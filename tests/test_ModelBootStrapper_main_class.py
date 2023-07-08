import unittest
import pandas as pd
import numpy as np
from ModelBootStrapper import ModelBootStrapper
from lightgbm import LGBMClassifier as clf


class ModelBootStrapperTestCase(unittest.TestCase):
    def test_class_estimator(self):
        with self.assertRaises(TypeError) as cm:
            ModelBootStrapper()
        self.assertEqual(
            str(cm.exception),
            "ModelBootStrapper.__init__() missing 1 required positional argument: 'estimator'",
        )

        with self.assertRaises(AssertionError) as cm:
            ModelBootStrapper(np)
        self.assertEqual(
            str(cm.exception),
            "estimator has to have fit, predict, and predict_proba methods",
        )

        with self.assertRaises(TypeError) as cm:
            ModelBootStrapper(clf)
        self.assertEqual(
            str(cm.exception), "estimator is not a class. Did you forget ()?"
        )

    def test_class_ci(self):
        with self.assertRaises(TypeError) as cm:
            ModelBootStrapper(clf(), ci=3)
        self.assertEqual(str(cm.exception), "ci has to be a float")

        with self.assertRaises(ValueError) as cm:
            ModelBootStrapper(clf(), ci=1.3)
        self.assertEqual(str(cm.exception), "ci has to be in the range (0, 1)")

    def test_class_n_boot(self):
        with self.assertRaises(ValueError) as cm:
            ModelBootStrapper(clf(), n_boot=0)
        self.assertEqual(str(cm.exception), "n_boot must be an integer > 0")

        with self.assertRaises(TypeError) as cm:
            ModelBootStrapper(clf(), n_boot=100.3)
        self.assertEqual(str(cm.exception), "n_boot must be an integer")

    def test_agg_func_callable(self):
        with self.assertRaises(AssertionError) as cm:
            ModelBootStrapper(clf(), agg_func=3)
        self.assertEqual(str(cm.exception), "agg_func has to be a callable")

    def test_class_n_folds(self):
        with self.assertRaises(ValueError) as cm:
            ModelBootStrapper(clf(), n_folds=0)
        self.assertEqual(str(cm.exception), "n_folds must be > 0")

        with self.assertRaises(TypeError) as cm:
            ModelBootStrapper(clf(), n_folds=3.3)
        self.assertEqual(str(cm.exception), "n_folds must be an integer")

    def test_fit_function(self):
        # should pass
        model = ModelBootStrapper(estimator=clf(), n_boot=3)
        X_train = pd.DataFrame(np.random.normal(size=(12, 2)))
        y_train = pd.Series([0, 1] * 6)
        model.fit(X_train, y_train)

        # unequal datasets
        with self.assertRaises(AssertionError) as cm:
            model.fit(X_train, y_train[:5])
        self.assertEqual(
            str(cm.exception), "X and y must have the same number of samples"
        )

        # y_train has only zeros
        y_train = pd.Series(np.zeros(12))
        with self.assertRaises(ValueError) as cm:
            model.fit(X_train, y_train)
        self.assertEqual(str(cm.exception), "y must include only [0, 1]")

    def test_predict_function(self):
        model = ModelBootStrapper(estimator=clf(), n_boot=3)
        n = 20
        X_train = pd.DataFrame(np.random.normal(size=(n, 2)))
        y_train = pd.Series([0, 1] * int(n / 2))

        with self.assertRaises(AssertionError) as cm:
            model.predict(X_train)
        self.assertEqual(str(cm.exception), "Please use `fit` before trying to predict")

        model.fit(X_train, y_train)
        preds = model.predict(X_train)
        self.assertEqual(preds.shape, (n, 3))


if __name__ == "__main__":
    unittest.main()
