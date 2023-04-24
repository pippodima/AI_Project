from Dataset import *

root_dir = '/Users/pippodima/Desktop/Uni/intelligenza artificiale/Progetto/new_tokenized_webkb'
load = True
def main():

    train_dataset = Dataset(root_dir, test=False, load=load)
    test_dataset = Dataset(root_dir, test=True, load=load)

    X_train, Y_train = train_dataset.get_X_Y()
    X_test, Y_test = test_dataset.get_X_Y()

if __name__ == "__main__":
    main()