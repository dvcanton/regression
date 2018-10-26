

import sklearn.learn_model import Ridge as sklearnRidge
import numpy as np

class Ridge:
    """ Ridge regression class """
    def __init__(self, alpha, implementation="sklearn"):
        if implementation == "sklearn":
            self.model = sklearnRidge(alpha=alpha, normalize=True)

    def fit(X, Y):
        self.model.fit(X, Y)

    def predict(X):
        self.model.predict(X)
