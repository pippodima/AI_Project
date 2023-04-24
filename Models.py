import numpy as np
class Bernoulli:
    def __int__(self):
        self.prior = []
        self.likelihood = []

    def train(self, X, y):
        X = np.where(X > 0, 1, 0)
        n_documents, vocab_size = np.shape(X)
        classes = np.unique(y)

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.prior.append(X_c.shape[0] / n_documents)
            self.likelihood.append((X_c.sum(axis=0) + 1) / (X_c.shape[0] + 2))

    def test(self, X):
        print('ciao')  # todo capire come legare eq 2 e 3 del pdf