# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 22:41:47 2019

@author: reynaldo.espana.rey

Train a deep learning model to write poems
First, train the spanish language through thousands of songs.
Then, train it to write poems trough transfer learning technique

source: https://www.analyticsvidhya.com/blog/2018/03/text-generation-using-python-nlp/

"""


# =============================================================================
# Libraries
# =============================================================================

import numpy as np
import pickle
import re
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Masking, Embedding
from keras.callbacks import EarlyStopping
from keras.models import load_model


# =============================================================================
# Input arguments
# =============================================================================
# is it a test? if True only 25 songs are used
TEST_MODE = False

# Where is the folder with all the corpus docs?
CORPUS_PATH = '../data/data_proccessed/NLP_data_poems_100-50'
# NLP
MAX_SEQ = 100
MIN_SEQ = 50

# Network params
MODEL_OUTPUT = '../models/'
MODEL_NAME = 'seq-100-50_layers-3_hn-600_batch-128'
EPOCHS = 15
BATCH_SIZE = 128
HIDDEN_NEURONS = 600


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

# build model
model = Sequential()
# construct inputs
# Mask parts of the lookback period that are all zeros (i.e., unobserved) so they don't skew the model
model.add(Masking(mask_value=-1, input_shape=(None, train_x.shape[2])))
# layer 1
model.add(LSTM(256, return_sequences=True, consume_less='gpu',
               recurrent_activation='sigmoid'))
model.add(Dropout(0.25))
# layer 2
model.add(LSTM(256, return_sequences=True, consume_less='gpu',
               recurrent_activation='sigmoid'))
model.add(Dropout(0.25))
# layer 3
model.add(LSTM(train_y.shape[1], return_sequences=False, consume_less='cpu',
               recurrent_activation='sigmoid', activation='softmax' ))
# Final layer
model.add(Dense(train_y.shape[1], activation='softmax'))
# compile
model.compile(loss='categorical_crossentropy',
              optimizer='RMSprop', # RMSprop, adam
              metrics=['accuracy'])

print(model.summary())


# =============================================================================
# Train with Songs
# =============================================================================
# callbacks
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.05, patience=2, verbose=1)

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
#class poem():
    
# Extract random sentence from corpus
def random_sentence(corpus, min_seq=64, max_seq=128):
    # use random sentence from corpus as seed
    doc = np.random.choice(corpus, 1)[0]
    text = doc['corpus']
    text_lines = text.split('\n')
    # select first lines till min_seq constraint is reached
    length = 0
    i=0
    sequence = str()
    while length <= min_seq:
        sequence = sequence + text_lines[i] + '\n'
        length = len(sequence)
        i+=1
    if length > max_seq:
        sequence = sequence[:max_seq]
    return text, sequence

# predict next character
def predict_next_char(sequence, n_to_char, char_to_n, model, max_seq=128, normalized_by=None, nans=0):
    # cut sequence into max length allowed   
    sequence = sequence[max(0, len(sequence)-max_seq):] 
    # do not cut the first word. always start with a full word
    # when the first word starts (look the first space or)
    first_word = np.argmax([x==' ' for x in sequence])
    # cut sequence
    sequence = sequence[first_word+1:]   
    # transform sentence to numeric
    sequence_encoded = [char_to_n[char] for char in sequence]
    sequence_encoded = np.reshape(sequence_encoded, (len(sequence), 1))
    # use sequence placeholder to fill in nans
    sequence_x = np.full((max_seq, 1), np.nan, dtype=np.float32)
    sequence_x[len(sequence_x)-len(sequence_encoded):] = sequence_encoded
    # Normalized
    if normalized_by is not None:
        sequence_x = (sequence_x + 1) / float(normalized_by)
    # replace nans by -1
    sequence_x[np.isnan(sequence_x)] = nans
    # reshape on tensor format
    sequence_x = np.reshape(sequence_x, (1, max_seq, 1))
    # predict next char
    pred_encoded = np.argmax(model.predict(sequence_x))
    pred_char = n_to_char[pred_encoded]
    return pred_char


def write_poem(seed, model,  n_to_char, char_to_n, max_seq=128, 
               normalized_by=None, nans=0, max_words=150):
    poem = seed
    # word count
    word_counter = len(re.findall(r'\w+', poem))
    # ends poem generator if max word is passed or poem end with final dot ($)
    while (word_counter < max_words) | (poem[-1]=='$') | (len(poem) < 1000):    
        # Prediction next character    
        next_char = predict_next_char(poem,
                                      n_to_char, char_to_n, model,
                                      max_seq, normalized_by)
        # append
        poem = poem + next_char
        # update word count
        word_counter = len(re.findall(r'\w+', poem))
    # add signature
    poem = poem + '\n\nEscrito por: AISP'
    return poem
        
    
    
    


# Generate poem

# random sentence to use as initial seed for the model
text, sequence = random_sentence(corpus, min_seq=64, max_seq=MAX_SEQ)
print('\nOriginal poem:\n\n', text)
print('\nSeed Sentence:\n\n', sequence)

poem = write_poem(sequence, model,  n_to_char, char_to_n, max_seq=MAX_SEQ, 
                  normalized_by=len(characters),  max_words=60)

print('\nThis is an AI poem:\n\n', poem)




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

