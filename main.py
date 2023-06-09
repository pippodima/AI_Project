from Dataset import *
from Models import *
from sklearn.metrics import accuracy_score

root_dir = '/Users/pippodima/Desktop/Uni/intelligenza artificiale/Progetto/new_tokenized_webkb'
load = True


def main():
    train_dataset = Dataset(root_dir, test=False, load=load)
    test_dataset = Dataset(root_dir, test=True, load=load)
    if not load:
        exit()
    X_train, Y_train = train_dataset.get_X_Y()
    X_test, Y_test = test_dataset.get_X_Y()

    bernoulliNaiveBayes = Bernoulli()
    bernoulliNaiveBayes.train(X_train, Y_train)
    Y_predBernoulli = bernoulliNaiveBayes.test(X_test)
    bernoulli_acc = accuracy_score(Y_test, Y_predBernoulli)
    print("Bernoulli: ", bernoulli_acc)

    multinomialNaiveBayes = Multinomial()
    multinomialNaiveBayes.train(X_train, Y_train)
    Y_predMultinomial = multinomialNaiveBayes.test(X_test)
    multinomial_acc = accuracy_score(Y_test, Y_predMultinomial)
    print("Multinomial: ", multinomial_acc)


if __name__ == "__main__":
    main()
