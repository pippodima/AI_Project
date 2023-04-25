from Dataset import *
from Models import *
from sklearn.metrics import accuracy_score


root_dir = '/Users/pippodima/Desktop/Uni/intelligenza artificiale/Progetto/new_tokenized_webkb'
load = True
def main():

    train_dataset = Dataset(root_dir, test=False, load=load)
    test_dataset = Dataset(root_dir, test=True, load=load)
    # exit()
    X_train, Y_train = train_dataset.get_X_Y()
    X_test, Y_test = test_dataset.get_X_Y()

    bernoulliNaiveBayes = Bernoulli()

    bernoulliNaiveBayes.train(X_train, Y_train)
    Y_pred = bernoulliNaiveBayes.test(X_test)


    print(accuracy_score(Y_test, Y_pred))
if __name__ == "__main__":
    main()
