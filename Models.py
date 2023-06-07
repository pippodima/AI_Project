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
            self.likelihood[i] = np.log((np.sum(X_c, axis=0) + 1) / (X_c.shape[0] + 2))
            self.not_likelihood[i] = np.log(1 + (np.sum(X_c + 1, axis=0) / (X_c.shape[0] + 2)))



    def test(self, X):
        X_ones = np.where(X > 0, 1, 0)
        n_documents, vocab_size = np.shape(X)
        temp = np.ones((n_documents, len(self.classes)))

        for i, c in enumerate(self.classes):
            temp[:, i] = np.sum((X_ones * self.likelihood[i]) + ((1-X_ones) * self.not_likelihood[i]), axis=1)
            temp[:, i] += self.prior[i]
        return self.classes[np.argmax(temp, axis=1)]


class Multinomial:
    def __init__(self):
        self.prior = None
        self.likelihood = None
        self.not_likelihood = None
        self.classes = None

    def train(self, X, y):
        n_documents, vocab_size = np.shape(X)
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        self.likelihood = np.zeros((n_classes, vocab_size))
        self.prior = np.zeros(n_classes)
        den_likelihood = np.sum(X)

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.likelihood[i] = np.log((np.sum(X_c, axis=0) + 1) / (den_likelihood + vocab_size))
            self.prior[i] = np.log(X_c.shape[0] / n_documents)


    def test(self, X):
        n_documents, vocab_size = np.shape(X)
        n_word_doc = np.sum(X, axis=1)
        len_doc, n_occ_doc = np.unique(n_word_doc, return_counts=True)
        log_prob_len_doc = np.log(n_occ_doc / n_documents)
        logfac_d = [sum([np.log(i) for i in range(1, j+1)]) for j in len_doc]
        logfac_X = np.zeros_like(X, dtype=np.float64)

        for d in range(n_documents):
            for w in range(vocab_size):
                for n in range(X[d, w]):
                    logfac_X[d, w] += np.log(n + 1)

        log_prob_len_doc += logfac_d
        probs = {}
        for i in range(len(len_doc)):
            probs[len_doc[i]] = log_prob_len_doc[i]

        final_probs = np.ones((n_documents, len(self.classes)))

        for i, c in enumerate(self.classes):
            final_probs[:, i] = np.sum(((X * self.likelihood[i]) - logfac_X), axis=1)

        for d in range(n_documents):
            for c in range(len(self.classes)):
                final_probs[d, c] += probs[n_word_doc[d]]

        final_probs += self.prior

        return self.classes[np.argmax(final_probs, axis=1)]



