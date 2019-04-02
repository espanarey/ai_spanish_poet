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
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from src.utils import build_model, random_sentence, write_poem

# =============================================================================
# Input arguments
# =============================================================================
# is it a test? if True only 25 songs are used
TEST_MODE = True

# Where is the folder with all the corpus docs?
CORPUS_PATH = './data/data_proccessed/NLP_data_poems_140'
# NLP
MAX_SEQ = 100

# Network params
MODEL_OUTPUT = './models/'
MODEL_NAME = 'seq-140_layers-3_encoding-256_batch-48'
ENCODING_OUT = 256
HIDDEN_UNITS = [1024, 1024, 768]
EPOCHS = 100
BATCH_SIZE = 48


# =============================================================================
# Load Data
# =============================================================================
# load data file
with open(CORPUS_PATH + '.pickle', 'rb') as file:
    data = pickle.load(file)

# unpack data dict
print('unpack:', data.keys())
# unpack elements
corpus = data['corpus']
words_mapping = data['words_mapping']
train_x = data['train_x']
train_y = data['train_y']
test_x = data['test_x']
test_y = data['test_y']

# unpack words mapping
print('unpack:', words_mapping.keys())
characters = words_mapping['characters']
n_to_char = words_mapping['n_to_char']
char_to_n = words_mapping['char_to_n']

# If it is a test mode just take first 10 batches of data
if TEST_MODE:
    print("TEST MODE: ON")
    # small data sample
    train_x = train_x[:BATCH_SIZE*5,:]
    train_y = train_y[:BATCH_SIZE*5,:]
    test_x = test_x[:BATCH_SIZE*2,:]
    test_y = test_y[:BATCH_SIZE*2,:]
    EPOCHS = 1
else:
    print("TEST MODE: OFF")

# validate data shape and size
print('Train data shape -', 'X:', train_x.shape, '- Y:', train_y.shape)
size = train_x.nbytes*1e-6 + train_y.nbytes*1e-6
size = print(int(size), 'Megabytes')


# =============================================================================
# LSTM Architecture
# =============================================================================
# create model
model = build_model(batch_size = BATCH_SIZE, 
                    encoding_dim = [len(n_to_char), ENCODING_OUT],
                    hidden_units = HIDDEN_UNITS,
                    optimizer= 'adam')


# =============================================================================
# Training
# =============================================================================
# callbacks
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5, verbose=1)

# save best
checkpoint = ModelCheckpoint(str(MODEL_OUTPUT + MODEL_NAME + '.h5'),
                             save_best_only=True, 
                             monitor='val_loss', verbose=1, period=3)

# samples to run. multiple of batch size
sample_train = (len(train_x)//BATCH_SIZE)*BATCH_SIZE
sample_test = (len(test_x)//BATCH_SIZE)*BATCH_SIZE

# Fit!
model_history = model.fit(train_x[:sample_train,:], train_y[:sample_train,:],
                          epochs = EPOCHS,
                          batch_size = BATCH_SIZE,
                          validation_data = (test_x[:sample_test,:], test_y[:sample_test,:]),
                          callbacks = [early_stop, checkpoint])

# Training history
print(model_history.history.keys())
# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# =============================================================================
# Test Example
# =============================================================================
# load model & weights 
# model to batch size 1 for prediction
model = build_model(batch_size = 1, 
                    encoding_dim = [len(n_to_char), ENCODING_OUT],
                    hidden_units = HIDDEN_UNITS,
                    optimizer= 'adam')

# load weights
model.load_weights(str(MODEL_OUTPUT + MODEL_NAME + '.h5'))

# Generate poem
print("\n\n---- Creative moment - I'm writting a poem")
# random sentence to use as initial seed for the model
text, sequence = random_sentence(corpus, min_seq=90, max_seq=MAX_SEQ)
print('\nOriginal poem:\n\n', text)
print('\nSeed Sentence:\n\n', sequence)

print('\n\n\nWriting poem ... \n\n')
poem = write_poem(sequence, model,  n_to_char, char_to_n,
                  max_seq=MAX_SEQ, max_words=160)

print('\nThis was an AI poem:\n\n', poem)