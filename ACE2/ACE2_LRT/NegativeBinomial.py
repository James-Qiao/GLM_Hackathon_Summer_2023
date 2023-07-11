import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import gamma

def add_const(X: pd.DataFrame):
    X['const'] = 1
    return X

# negative binomial regression
class NB_Reg():
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.params = np.ones((X.shape[1]), dtype=float)
        # self.params[:-1] = coef
        # self.params[-1] = phi
        # phi = 1/alpha, dispersion parameter
        self.phi = 1

        self.optimize_record_w = []
        self.optimize_record_phi = []
        self.optimize_record = []

    def log_likelihood(self, params_phi): #log likelihood of negative binomial distribution
        mu = np.exp(np.dot(self.X, params_phi[:-1]))
        phi = params_phi[-1]
        Y = self.Y
        ll = np.sum(np.log((gamma(Y+phi)/((gamma(Y+1)*gamma(phi))))*((phi/(phi+mu))**phi)*((mu/(phi+mu))**Y)))
        # return negative log likelihood
        return np.sum(ll) * -1
    
    def log_likelihood_w(self, params): #log likelihood of negative binomial distribution
        mu = np.exp(np.dot(self.X, params))
        Y = self.Y
        phi = self.phi
        ll = np.sum(np.log((gamma(Y+phi)/((gamma(Y+1)*gamma(phi))))*((phi/(phi+mu))**phi)*((mu/(phi+mu))**Y)))
        # return negative log likelihood
        return np.sum(ll) * -1
    
    def log_likelihood_phi(self, phi): #log likelihood of negative binomial distribution
        mu = np.exp(np.dot(self.X, self.params))
        Y = self.Y
        ll = np.sum(np.log((gamma(Y+phi)/((gamma(Y+1)*gamma(phi))))*((phi/(phi+mu))**phi)*((mu/(phi+mu))**Y)))
        # return negative log likelihood
        return np.sum(ll) * -1

    def fit_w(self):
        w = self.params
        w = optimize.minimize(self.log_likelihood_w, w, method='BFGS')
        self.params = w.x
        self.optimize_record_w = w
        return self

    def fit_phi(self):
        phi = self.phi
        phi = optimize.minimize(self.log_likelihood_phi, phi, method='BFGS')
        self.phi = phi.x
        self.optimize_record_phi = phi
        return self
    
    def fit(self):
        params = self.params
        phi = self.phi
        params_phi = np.append(params, phi)
        bounds = [(None, None)] * len(params) + [(0, None)]
        params_phi = optimize.minimize(self.log_likelihood, params_phi, method='powell', bounds=bounds)
        self.params = params_phi.x[:-1]
        self.phi = params_phi.x[-1]
        self.optimize_record = params_phi
        return self
    
    def predict(self, X):
        return np.exp(np.dot(X, self.params))