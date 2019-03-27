# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 22:41:47 2019

@author: reynaldo.espana.rey

Train a deep learning model to write poems
First, train the spanish language through thousands of songs.
Then, train it to write poems trough transfer learning technique

source: https://www.analyticsvidhya.com/blog/2018/03/text-generation-using-python-nlp/


watch nvidia-smi

screen -S sessionname (assign name to session)
screen -r (list of session)
screen -x (resume)
"""


# =============================================================================
# Libraries
# =============================================================================

# TODO: create env and .yml
import numpy as np
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt
from sklearn.utils import shuffle # install
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Masking
from keras.callbacks import EarlyStopping
from keras.models import load_model


# =============================================================================
# Input arguments
# =============================================================================
# is it a test? if True only 25 songs are used
TEST_MODE = False

# Where is the folder with all the corpus docs?
CORPUS_PATH = '../data/data_proccessed/NLP_data_poems_120-40'
# NLP
MAX_SEQ = 120
MIN_SEQ = 40

# Network params
MODEL_OUTPUT = '../models/'
MODEL_NAME = 'seq-120_layers-3_hn-512'
EPOCHS = 5
BATCH_SIZE = 64
HIDDEN_NEURONS = 512


# =============================================================================
# Load Data
# =============================================================================

# load data file
with open(CORPUS_PATH + '.pickle', 'rb') as file:
    data = pickle.load(file)

# unpack data dict
print('unpack:', data.keys())
for item in data.keys(): exec(item + " = eval('data[item]') ")

# unpack words mapping
print('unpack:', words_mapping.keys())
for item in words_mapping.keys(): exec(item + " = eval('words_mapping[item]') ")


# If it is a test mode just take first 10 batches of data
if TEST_MODE:
    print("TEST MODE: ON")
    # small data sample
    train_x = train_x[:BATCH_SIZE*20,:,:]
    train_y = train_y[:BATCH_SIZE*20,:]
    test_x = test_x[:BATCH_SIZE*5,:,:]
    test_y = test_y[:BATCH_SIZE*5,:]
else:
    print("TEST MODE: OFF")

# validate data shape and size
print('Train data shape -', 'X:', train_x.shape, '- Y:', train_y.shape)
size = train_x.nbytes*1e-6 + train_y.nbytes*1e-6
size = print(int(size), 'Megabytes')


# =============================================================================
# LSTM Architecture
# =============================================================================
print('\n---\nModel network')

model = Sequential()
# Mask parts of the lookback period that are all zeros (i.e., unobserved) so they don't skew the model
model.add(Masking(mask_value=-1., input_shape=(train_x.shape[1], train_x.shape[2])))
# layer 1
model.add(LSTM(HIDDEN_NEURONS, return_sequences=True))
model.add(Dropout(0.2))
# layer 2
model.add(LSTM(HIDDEN_NEURONS, return_sequences=True)))
model.add(Dropout(0.2))
# layer 3
model.add(LSTM(HIDDEN_NEURONS))
model.add(Dropout(0.2))
# Final layer
model.add(Dense(train_y.shape[1], activation='softmax'))
# compile
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

# TODO: add extra dense layer before last layer

# =============================================================================
# Train with Songs
# =============================================================================
# callbacks
# TODO: val_set
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=2, verbose=1)

# Fit!
model_history = model.fit(train_x, train_y,
                          epochs = EPOCHS,
                          batch_size = BATCH_SIZE,
                          validation_data = (test_x, test_y),
                          callbacks = [early_stop])

# Training history
print(model_history.history.keys())
# summarize history for loss
plt.plot(model_history.history['loss'])
#plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()

## save model
model.save(str(MODEL_OUTPUT + MODEL_NAME + '.h5'))
print('Model saved in: ', str(MODEL_OUTPUT + MODEL_NAME + '.h5'))


# =============================================================================
# Test Example
# =============================================================================
# Extract random sentence from corpus
def random_sentence(corpus, min_seq=64, max_seq=128):
    # use random sentence from corpus as seed
    doc = np.random.choice(corpus, 1)[0]
    text = doc['corpus']
    # random character to start
    # TODO: do not cut words
    idx_from = max(0, int(np.random.uniform(0, len(text)) - min_seq))
    # length of sequence to use as seed
    seq_length = int(np.random.uniform(min_seq, max_seq))
    # initial sentence
    sequence = text[idx_from:min(len(text), idx_from+seq_length)]
    return sequence

# predict next character
def predict_next_char(sequence, n_to_char, char_to_n, model, max_seq=128, normalized_by=None, nans=-1.):
    # transform sentence to numeric
    sequence_encoded = [char_to_n[char] for char in sequence]
    sequence_encoded = np.reshape(sequence_encoded, (len(sequence), 1))
    # use sequence placeholder to fill in nans
    sequence_x = np.full((max_seq, 1), np.nan, dtype=np.float32)
    sequence_x[len(sequence_x)-len(sequence_encoded):] = sequence_encoded
    # Normalized
    if normalized_by is not None:
        sequence_x = sequence_x / float(normalized_by)
    # replace nans by -1
    sequence_x[np.isnan(sequence_x)] = nans
    # reshape on tensor format
    sequence_x = np.reshape(sequence_x, (1, max_seq, 1))
    # predict next char
    pred_encoded = np.argmax(model.predict(sequence_x))
    pred_char = n_to_char[pred_encoded]
    return pred_char





# TODO: predict entire text

# string_mapped.append(pred_index)
# string_mapped = string_mapped[1:len(string_mapped)]


# random sentence to use as initial seed for the model
sequence = random_sentence(corpus, min_seq=64, max_seq=MAX_SEQ)
print('sentence:\n', sequence)
next_char = predict_next_char(sequence,
                              n_to_char, char_to_n, model,
                              max_seq=MAX_SEQ, normalized_by=len(characters))
print('\nnext char:', next_char)




# =============================================================================
# Transfer Learning with poems (separate script)
# =============================================================================
# load
model = load_model(str(MODEL_OUTPUT + MODEL_NAME + '.h5'))

# Train
# Save model
# test model


# =============================================================================
# Final Poem (separate script)
# =============================================================================
# load model
# input data structure
# 'te escribo el siguiente poema\naunque no sea un humano.\n'
# output
