import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras import utils

def preprocess_x(text, seqlen = 100):
    # get the number of maximum words in a review
    num_words = max([len(t.split()) for t in text], key=int)

    # instansiate and fit tokenizer
    tokenizer = Tokenizer(num_words = num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(text)

    # get x 
    x = tokenizer.texts_to_sequences(text)
    x = utils.pad_sequences(x, maxlen=seqlen, padding='post', truncating='post')
    print(f'Done: x has shape {x.shape}')
    return x
