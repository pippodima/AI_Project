import numpy as np
class Bernoulli:
    def __int__(self):
        self.prior = None
        self.likelihood = None

    def train(self, X, y):
        X_ones = np.where(X > 0, 1, 0)
        n_documents, vocab_size = np.shape(X)
        classes = np.unique(y)

        for i, c in enumerate(classes):
            X_c = X_ones[y == c]
            self.prior.append(X_c.shape[0] / n_documents)
            self.likelihood.append((X_c.sum(axis=0) + 1) / (X_c.shape[0] + 2))

        self.likelihood = np.reshape((len(classes), vocab_size))
        self.prior = np.reshape((len(classes), 1))

    def test(self, X):
        X_ones = np.where(X > 0, 1, 0)
        n_documents, vocab_size = np.shape(X)

        self.likelihood = (X_ones * self.likelihood) + (1-X_ones) * (1-self.likelihood)




        print('ciao')  # todo capire come legare eq 2 e 3 del pdf