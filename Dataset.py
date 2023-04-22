import os
import numpy as np

categories = ["course", "department", "faculty", "other", "project", "staff", "student"]
universities = ['cornell', 'misc', 'texas', 'washington', 'wisconsin']

class Dataset:
    def __init__(self, root_dir, ):
        self.word_to_index = {}
        self.index_to_word = {}
        self.labels = []
        self.vocab_size = 0


        self._build_vocab(root_dir)
        self.word_counts = np.zeros((documents, self.vocab_size)) #todo trovare il numero di documenti per creare la matrice




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
            for uni in universities:
                folder_path = f'{root_dir}/{category}/{uni}'
                for element in os.listdir(folder_path):
                    file_path = f'{folder_path}/{element}'
                    if element.endswith('.txt'):
                        with open(file_path) as file:
                            for line in file:
                                for word in line.split():

