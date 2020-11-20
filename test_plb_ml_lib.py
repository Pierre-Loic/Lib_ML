import pytest
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
import plb_ml_lib as pml

@pytest.mark.parametrize("solver", ["normal_eq", "pseudo_inv", "ols", "gd", "sgd"])
def test_LNNp_normal(solver):
    reg_ref = LinearRegression()
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]]) 
    y = np.dot(X, np.array([1, 2])) + 3*np.random.rand() # y = 1 * x_0 + 2 * x_1 + 3
    reg_ref.fit(X, y)
    reg_test = pml.LinearRegressionNp(solver)
    reg_test.fit(X, y)
    np.testing.assert_array_almost_equal(reg_ref.coef_, reg_test.coef_, decimal=2)
    np.testing.assert_array_almost_equal(reg_ref.intercept_, reg_test.intercept_, decimal=2)
    np.testing.assert_array_almost_equal(reg_ref.predict(np.array([[3, 5]])), reg_test.predict(np.array([[3, 5]])), decimal=2)

@pytest.mark.parametrize("alpha", [0.001, 0.01, 0.1, 1, 10, 100, 1000])
def test_RidgeNp_normal(alpha):
    n_samples, n_features = 10000, 5
    y_train = np.random.randn(n_samples)
    X_train = np.random.randn(n_samples, n_features)
    ridge_ref = Ridge(alpha=alpha)
    ridge_test = pml.RidgeNp("normal_eq", alpha=alpha)
    ridge_ref.fit(X_train, y_train)
    ridge_test.fit(X_train, y_train)
    X_test = np.random.randn(n_samples, n_features)
    np.testing.assert_array_almost_equal(ridge_ref.coef_, ridge_test.coef_, decimal=3)
    np.testing.assert_array_almost_equal(ridge_ref.intercept_, ridge_test.intercept_, decimal=3)
    np.testing.assert_array_almost_equal(ridge_ref.predict(X_test), ridge_test.predict(X_test), decimal=3)