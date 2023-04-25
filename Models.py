import numpy as np
class Bernoulli:
    def __init__(self):
        self.prior = None
        self.likelihood = None
        self.not_likelihood = None
        self.classes = None

    def train(self, X, y):
        X_ones = np.where(X > 0, 1, 0)
        n_documents, vocab_size = np.shape(X)
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        self.likelihood = np.zeros((n_classes, vocab_size))
        self.not_likelihood = np.zeros((n_classes, vocab_size))
        self.prior = np.zeros(n_classes)

        for i, c in enumerate(self.classes):
            X_c = X_ones[y == c]
            self.prior[i] = np.log(X_c.shape[0] / n_documents)
            self.likelihood[i] = np.log((X_c.sum(axis=0) + 1) / (X_c.shape[0] + 2))
            self.not_likelihood[i] = np.log(1 - ((X_c.sum(axis=0) + 1) / (X_c.shape[0] + 2)))


    def test(self, X):
        X_ones = np.where(X > 0, 1, 0)
        n_documents, vocab_size = np.shape(X)
        temp = np.zeros((n_documents, len(self.classes)))

        for i, c in enumerate(self.classes):
            temp[:, i] = np.prod((X_ones * self.likelihood[i]) + ((1-X_ones) * self.not_likelihood[i]), axis=1)
            # temp[:, i] *= self.prior[i]
        den = np.sum(temp, axis=1)
        temp = temp/den
        return self.classes[np.argmax(temp, axis=1)]

np.random.shuffle()