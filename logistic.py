import sklearn.datasets
import numpy as np

class Logistic:
    """ Logistic class """
    def __init__(self,  lr=0.01, num_iter=100000, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.verbose = verbose

    def signmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    # Log loss / Cross Entropy Loss
    # -(y log p + (1-y)log(1-p))
    def loss(self, h, y):
        return (-y * np.log(h) + (1-y) * np.log(1-h)).mean()


    def predict_prob(self, X):
        h = signmoid(np.dot(X, self.theta))
        return h

    def predit(self, X, threashold):
        return self.predict_prob(X) >= threashold

    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.signmoid(z)
            gradient = np.dot(X.T, (h - y) / y.shape[0])
            self.theta -= self.lr * gradient

            if(self.verbose == True and i % 10000 == 0):
                # print("theta:")
                # print(self.theta)
                print("loss: %.2f" % self.loss(h, y))


model = Logistic(lr=0.1, num_iter=200000, verbose=True)

iris = sklearn.datasets.load_iris()
X = iris.data[:, :2]
y = (iris.target != 0) * 1

model.fit(X, y)
