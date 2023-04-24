import os
from bs4 import BeautifulSoup
import re
from StopWords import get_stopword
from nltk.tokenize import word_tokenize

root_dir = '/Users/pippodima/Desktop/Uni/intelligenza artificiale/Progetto'
mime_pattern = re.compile(r'^.*?(<!doctype|<html|<head|<title)', re.DOTALL | re.IGNORECASE)
mystopwords = get_stopword()
categories = ["course", "department", "faculty", "other", "project", "staff", "student"]
universities = ['cornell', 'misc', 'texas', 'washington', 'wisconsin']
error = 0
for category in categories:
    for uni in universities:
        directory = f'{root_dir}/webkb/{category}/{uni}'
        for element in os.listdir(directory):
            file_directory = directory + f'/{element}'
            with open(file_directory) as file:
                try:
                    contents = file.read()
                except:
                    print(f"errore in file {file_directory}")
                    error += 1

                contents = mime_pattern.sub(r'\1', contents)  # mime message
                soup = BeautifulSoup(contents, 'html.parser')
                raw = soup.getText()
                raw = re.sub(r'[^\w\s]', ' ', raw)  # punctuation
                raw = re.sub(r'\d+', '', raw)  # numbers
                raw = re.sub('\s+', ' ', raw).strip()  # double spaces
                word_tokens = word_tokenize(raw)
                words = [word.lower() for word in word_tokens if word.lower() not in mystopwords and len(word.lower()) > 2]


                output_directory = f'{root_dir}/new_tokenized_webkb/{category}/{uni}'
                os.makedirs(output_directory, exist_ok=True)
                output_file_name = f'{output_directory}/{element}.txt'
                with open(output_file_name, 'w') as output:
                    output.write(' '.join(words))
print(error)