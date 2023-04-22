def get_stopword():
    homepage = ['home', 'page', 'homepage']
    with open("my_stopwords/stopwords.txt", "r") as file:
        lst = [s.strip() for s in file.readlines()]
    lst.extend(homepage)
    return lst
