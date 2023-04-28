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

    sizes = [0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1]

    for size in sizes:
        X_train_new = get_different_vocabulary(X_train, Y_train, size)
        X_test_new = get_different_vocabulary(X_test, Y_test, size)
        bernoulliNaiveBayes = Bernoulli()
        bernoulliNaiveBayes.train(X_train_new, Y_train)
        Y_pred_Bernoulli = bernoulliNaiveBayes.test(X_test_new)
        print('Bernoulli: ', accuracy_score(Y_test, Y_pred_Bernoulli), 'size: ', size)

        multinomialNaiveBayes = Multinomial()
        multinomialNaiveBayes.train(X_train_new, Y_train)
        Y_pred_Multinomial = multinomialNaiveBayes.test(X_test_new)
        print('Multnomial: ', accuracy_score(Y_test, Y_pred_Multinomial), 'size: ', size)
        print()


def get_different_vocabulary(X, y, size):
    X_ones = np.where(X > 0, 1, 0)
    n_documents, vocab_size = np.shape(X)
    p_f = (np.sum(X_ones, axis=0)) / n_documents
    for i, c in enumerate(np.unique(y)):
        X_c = X[y == c]
        p_c = X_c.shape[0]/n_documents
        p_c_f = (np.sum(X_c, axis=0)) / n_documents
    k = 0.5
    p_c += k
    p_f += k
    p_c_f += k
    mutual_inf = p_c_f * np.log(p_c_f/(p_c*p_f))
    size = int(vocab_size * size)
    top_mutual_word_indices = np.argsort(mutual_inf)[::1]
    top_mutual_word_indices = top_mutual_word_indices[:size]
    X_new = X[:, top_mutual_word_indices]
    return X_new


if __name__ == "__main__":
    main()
