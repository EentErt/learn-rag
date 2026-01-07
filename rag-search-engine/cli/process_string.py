from nltk.stem import PorterStemmer
import string

def process_string(input):
    result = input.lower()

    result = remove_punctuation(result)

    result = tokenize(result)

    result = remove_stop_words(result)

    result = stem(result)

    return result

def remove_punctuation(input):
    result = ""
    for char in input:
        if char not in string.punctuation:
            result += char
    return result

def tokenize(input):
    word_list = input.split()
    result = []
    for word in word_list:
        if word != "":
            result.append(word)
    return result

def remove_stop_words(input):
    with open("data/stopwords.txt", "r") as file:
        stop_words = file.read().splitlines()

    result = []
    for word in input:
        if word not in stop_words:
            result.append(word)

    return result

def stem(input):
    stemmer = PorterStemmer()
    result = []
    for word in input:
        stem_word = stemmer.stem(word)
        result.append(stem_word)
    return result
