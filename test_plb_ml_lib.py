import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
import plb_ml_lib as pml

def test_LNNp_normal():
    reg_ref = LinearRegression()
    reg_test = pml.LinearRgressionNp()
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]]) 
    y = np.dot(X, np.array([1, 2])) + 3*np.random.rand() # y = 1 * x_0 + 2 * x_1 + 3
    reg_ref.fit(X, y)
    reg_test.fit(X, y)
    np.testing.assert_array_almost_equal(reg_ref.coef_, reg_test.coef_)
    np.testing.assert_array_almost_equal(reg_ref.intercept_, reg_test.intercept_)
    np.testing.assert_array_almost_equal(reg_ref.predict(np.array([[3, 5]])), reg_test.predict(np.array([[3, 5]])))