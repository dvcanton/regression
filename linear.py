
from sklearn.linear_model import LinearRegression as sklearnLinearRegression
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    """ Ridge regression class """
    def __init__(self, data, power=2, implementation="sklearn"):
        self.power=power
        self.data
        if implementation == "sklearn":
            self.model = sklearnLinearRegression(normalize=True)

    def fit(X, Y):
        self.model.fit(X, Y)

    def predict(X):
        self.model.predict(X)

    def plot():
        models_to_plot = {1:231,3:232,6:233,9:234,12:235,15:236}
        plt.subplot(models_to_plot[self.power])
        plt.tight_layout()
        plt.plot(data['x'],y_pred)
        plt.plot(data['x'],data['y'],'.')
        plt.title('Plot for power: %d'%self.power)
