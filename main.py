import numpy as np
import matplotlib.pyplot as plt
from Dataset import *
from Models import *
from sklearn.metrics import accuracy_score
import sklearn as sl


root_dir = '/Users/pippodima/Desktop/Uni/intelligenza artificiale/Progetto/new_tokenized_webkb'
load = True
def main():

    train_dataset = Dataset(root_dir, test=False, load=load)
    test_dataset = Dataset(root_dir, test=True, load=load)
    # exit()
    X_train, Y_train = train_dataset.get_X_Y()
    X_test, Y_test = test_dataset.get_X_Y()

    sizes = [1]  # [0.0 001, 0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1]
    bernoulli_acc = []
    multinomial_acc = []
    vocab_len = []

    for size in sizes:

        X_train_new, vocab_size = get_different_vocabulary(X_train, size)
        X_test_new, vocab_size = get_different_vocabulary(X_test, size)
        vocab_len.append(vocab_size)

        bernoulliNaiveBayes = Bernoulli()
        bernoulliNaiveBayes.train(X_train_new, Y_train)
        Y_pred_Bernoulli = bernoulliNaiveBayes.test(X_test_new)
        print('Bernoulli: ', accuracy_score(Y_test, Y_pred_Bernoulli), 'size: ', size)
        bernoulli_acc.append(accuracy_score(Y_test, Y_pred_Bernoulli))

        multinomialNaiveBayes = Multinomial()
        multinomialNaiveBayes.train(X_train_new, Y_train)
        Y_pred_Multinomial = multinomialNaiveBayes.test(X_test_new)
        print('Multnomial: ', accuracy_score(Y_test, Y_pred_Multinomial), 'size: ', size)
        multinomial_acc.append(accuracy_score(Y_test, Y_pred_Multinomial))
        print()

    plt.plot(vocab_len, bernoulli_acc, 'r', marker='o', linestyle='-')
    plt.plot(vocab_len, multinomial_acc, 'b', marker='o', linestyle='-')
    plt.xscale('log')
    plt.grid()

    plt.xlabel('vocabulary size')
    plt.ylabel('accuracy')
    plt.ylim(0, 1)

    plt.show()




def get_different_vocabulary(X, size):
    n_documents, vocab_size = np.shape(X)
    word_occurences = np.sum(X, axis=0)
    vocab_len = int(size*vocab_size)
    top_indices = np.argsort(word_occurences)[-vocab_len:]
    X_new = X[:, top_indices[::-1]]
    return X_new, vocab_len



if __name__ == "__main__":
    main()
