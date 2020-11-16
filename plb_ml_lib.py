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
        elif self.solver == "ols":
            self._fit_ols(X, y)
        elif self.solver == "gd":
            self._fit_gd(X, y)
        elif self.solver == "sgd":
            pass
        elif self.solver == "bgd":
            pass
        else:
            print(f"Solver {self.solver} non reconnu")
            return
        self._update_parameters()
    
    def predict(self, X):
        X_1 = self._add_constant(X)
        return X_1.dot(self.theta)

    def _add_constant(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]

    # Fit functions
    def _fit_normal(self, X, y):
        X_1 = self._add_constant(X)
        self.theta = np.linalg.inv(X_1.T.dot(X_1)).dot(X_1.T.dot(y))

    def _fit_pseudo_inv(self, X, y):
        X_1 = self._add_constant(X)
        self.theta = np.linalg.pinv(X_1).dot(y)

    def _fit_ols(self, X, y):
        X_1 = self._add_constant(X)
        self.theta = np.linalg.lstsq(X_1, y, rcond=1e-6)[0]

    def _fit_gd(self, X, y, learning_rate=0.01, n_iter=10000):
        X_1 = self._add_constant(X)
        y = y.reshape(-1,1)
        self.theta = np.random.randn(X_1.shape[1], 1)
        for i in range(n_iter):
            gradient = (2/X_1.shape[0])*X_1.T.dot(X_1.dot(self.theta)-y)
            self.theta = self.theta - learning_rate*gradient
        self.theta = self.theta.flatten()

    def _update_parameters(self):
        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]

    def _check_input(self, X, y):
        """ Check the size and the types of the data """
        pass

