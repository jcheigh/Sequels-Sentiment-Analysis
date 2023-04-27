# data
import numpy as np
import pandas as pd
import re
import pickle

# keras
from keras.preprocessing.text import Tokenizer
from keras import utils

# nltk
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# preprocess data for LSTM given a dataframe
def preprocess_x(text, seqlen = 100, verbose=True):
    # get the number of maximum words in a review
    num_words = max([len(t.split()) for t in text], key=int)

    # load tokenizer
    with open('./tokenizer/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # get x 
    x = tokenizer.texts_to_sequences(text)
    x = utils.pad_sequences(x, maxlen=seqlen, padding='post', truncating='post')
    if verbose:
        print(f'Done: x has shape {x.shape}')

    # get word index
    word_index = tokenizer.word_index
    if verbose:
        print(f"Found {len(word_index)} words")
    return x, word_index

# strip stop words and other extraneous words
def parse_text(text, verbose=False):
    text = re.sub("[^a-zA-Z]", ' ', text)
    if verbose:
        print('removed punctuation/numbers')

    text = text.lower().split()
    if verbose:
        print('make lowercase and split')

    stop_words = set(stopwords.words("english"))
    text = [w for w in text if w not in stop_words]
    if verbose:
        print('removed stop words')

    return ' '.join(text)

def hi():
    print(set(stopwords.words("english")))

def parse_texts(texts):
    new_texts = []
    for t in texts:
        new_texts.append(parse_text(t))
    return new_texts