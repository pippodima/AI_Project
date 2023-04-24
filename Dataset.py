import os
import numpy as np

categories = ["course", "department", "faculty", "other", "project", "staff", "student"]
universities = ['cornell', 'misc', 'texas', 'washington', 'wisconsin']
train_universities = ['cornell', 'misc', 'texas', 'washington']
test_universities = ['wisconsin']

class Dataset:
    def __init__(self, root_dir, test, load):
        self.word_to_index = {}
        self.index_to_word = {}
        self.labels = []
        self.word_counts = []
        self.vocab_size = 0
        self.n_documents = 0

        self.output_directory = 'test' if test else 'train'
        self.universities = test_universities if test else train_universities


        if not load:
            self._build_vocab(root_dir)
            self._count_words(root_dir)
            self.word_counts = np.reshape(self.word_counts, (self.n_documents, self.vocab_size))
            self.labels = np.reshape(self.labels, (self.n_documents, 1))
            os.makedirs(self.output_directory, exist_ok=True)
            np.save(self.output_directory + '/word_counts', self.word_counts)
            np.save(self.output_directory + '/labels', self.labels)
        else:
            self.word_counts = np.load(self.output_directory + '/word_counts.npy')
            self.labels = np.load(self.output_directory + '/labels.npy')



    def get_X(self):
        return self.word_counts

    def get_Y(self):
        return self.labels

    def get_X_Y(self):
        return self.word_counts, self.labels

    def _build_vocab(self, root_dir):
        for category in categories:
            for uni in universities:
                folder_path = f'{root_dir}/{category}/{uni}'
                for element in os.listdir(folder_path):
                    file_path = f'{folder_path}/{element}'
                    if element.endswith('.txt'):
                        with open(file_path) as file:
                            for line in file:
                                for word in line.split():
                                    if word not in self.word_to_index:
                                        index = len(self.word_to_index)
                                        self.word_to_index[word] = index
                                        self.index_to_word[index] = word

        self.vocab_size = len(self.word_to_index)

    def _count_words(self, root_dir):
        for category in categories:
            for uni in self.universities:
                folder_path = f'{root_dir}/{category}/{uni}'
                for element in os.listdir(folder_path):
                    file_path = f'{folder_path}/{element}'
                    if element.endswith('.txt'):
                        word_counts_doc = [0] * self.vocab_size
                        with open(file_path) as file:
                            for line in file:
                                for word in line.split():
                                    word_counts_doc[self.word_to_index[word]] += 1
                    self.word_counts.append(word_counts_doc)
                    self.labels.append(category)
        self.n_documents = len(self.labels)
