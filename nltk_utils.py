import nltk
import numpy as np
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    # TODO for implitation
    tokenized_sentence = list(set([stem(w) for w in tokenized_sentence]))
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag


if __name__ == '__main__':
    a = "How long does shipping take?".split()
    words = "hi hello how I you bye thank cool".split()
    bog = bag_of_words(a, words)
    print(bog)