import numpy as np

class LinearRegressionPy:
    pass

class LinearRegressionNp:
    
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
            self._fit_sgd(X, y)
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

    def _fit_sgd(self, X, y, t0=800, lr0=0.1, n_epochs=500):
        X_1 = self._add_constant(X)
        y = y.reshape(-1,1)
        self.theta = np.random.randn(X_1.shape[1], 1)
        for epoch in range(n_epochs):
            random_index = np.random.randint(X_1.shape[0])
            X_i = X_1[random_index:random_index+1]
            y_i = y[random_index:random_index+1]
            gradient = 2*X_i.T.dot(X_i.dot(self.theta)-y_i)
            learning_rate = lr0*(t0/(t0+epoch))
            self.theta = self.theta - learning_rate*gradient
        self.theta = self.theta.flatten()

    def _fit_bgd(self, X, y, learning_rate=0.01, n_iter=10000):
        pass

    def _update_parameters(self):
        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]

class RidgeNp:

    def __init__(self, solver="normal_eq", alpha=1):
        self.solver = solver
        self.alpha = alpha
        self.theta = None
        self.intercept_ = None
        self.coef_ = None

    def fit(self, X, y):
        if self.solver == "normal_eq":
            self._fit_normal(X, y)
        elif self.solver == "gd":
            pass
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

    def _fit_normal(self, X, y):
        X_1 = self._add_constant(X)
        self.theta = np.linalg.inv(X_1.T.dot(X_1)+self.alpha*np.identity(X_1.shape[1])).dot(X_1.T.dot(y))

    def _add_constant(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]

    def _update_parameters(self):
        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]