import numpy as np

class LinearRegressionPy:
    pass

class LinearRgressionNp:
    
    def __init__(self, solver="normal_eq"):
        self.solver = solver
        self.theta = None
        self.intercept_ = None
        self.coef_ = None

    def fit(self, X, y):
        if self.solver == "normal_eq":
            self._fit_normal(X, y)
        elif self.solver == "pseudo_inv":
            self._fit_pseudo_inv(X, y)
        else:
            print(f"Solver {self.solver} non reconnu")
            return
        self._update_parameters()
    
    def predict(self, X):
        X_1 = np.c_[np.ones((X.shape[0], 1)), X]
        return X_1.dot(self.theta)

    def _fit_normal(self, X, y):
        X_1 = np.c_[np.ones((X.shape[0], 1)), X]
        self.theta = np.linalg.inv(X_1.T.dot(X_1)).dot(X_1.T.dot(y))

    def _fit_pseudo_inv(self, X, y):
        pass

    def _update_parameters(self):
        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]

    def _check_input(self, X, y):
        """ Check the size and the types of the data """
        pass

