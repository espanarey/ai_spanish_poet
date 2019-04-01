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
from keras.layers import Dense, LSTM, Dropout, Masking
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from tensorflow.keras.losses import sparse_categorical_crossentropy


# =============================================================================
# Input arguments
# =============================================================================
# is it a test? if True only 25 songs are used
TEST_MODE = False

# Where is the folder with all the corpus docs?
CORPUS_PATH = './data/data_proccessed/NLP_data_love_160_seq2seq'
# NLP
MAX_SEQ = 160

# Network params
MODEL_OUTPUT = './models/'
MODEL_NAME = 'seq-160_layers-3_encoding-256_batch-128_seq2seq'
ENCODING = 256
EPOCHS = 15
BATCH_SIZE = 16



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
    train_x = train_x[:BATCH_SIZE*20,:]
    train_y = train_y[:BATCH_SIZE*20,:]
    test_x = test_x[:BATCH_SIZE*5,:]
    test_y = test_y[:BATCH_SIZE*5,:]
    EPOCHS = 1
else:
    print("TEST MODE: OFF")

# validate data shape and size
print('Train data shape -', 'X:', train_x.shape, '- Y:', train_y.shape)
size = train_x.nbytes*1e-6 + train_y.nbytes*1e-6
size = print(int(size), 'Megabytes')


# =============================================================================
# Load Model and Freeze layers
# =============================================================================
# Model loss
def loss(y_true, y_pred):
    return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

# load model & weights 
model = load_model(str(MODEL_OUTPUT + MODEL_NAME + '.h5'), custom_objects={'loss': loss})

# Model Summary 
print('\n---\nModel network')
print(model.summary())

# Freeze all but last layer
print('Frezz last 2 layers')
for layer in model.layers[:-2]:
    layer.trainable=False

for i,layer in enumerate(model.layers):    
  print('Layer id:', i, layer.name, 'Trainable:', layer.trainable)
  

# =============================================================================
# Train with love poems
# =============================================================================
# callbacks
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
checkpoint = ModelCheckpoint(str(MODEL_OUTPUT + MODEL_NAME + '_TL' +'.h5'),
                             monitor='val_loss', verbose=1, save_best_only=True)

# Fit!
model_history = model.fit(train_x, train_y,
                          epochs = EPOCHS,
                          batch_size = BATCH_SIZE,
                          validation_data = (test_x, test_y),
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
def predict_next_char(sequence, n_to_char, char_to_n, model, max_seq=128):
    # cut sequence into max length allowed   
    sequence = sequence[max(0, len(sequence)-max_seq):] 
    # transform sentence to numeric
    sequence_encoded = np.array([char_to_n[char] for char in sequence])
    pred_encoded = model.predict(sequence_encoded)  
    # last prediction in sequence
    pred = pred_encoded[-1][0]
    # from log probabilities to normalized probabilities
    pred_prob = np.exp(pred)/np.sum(np.exp(pred))
    # next char
    pred_char = np.argmax(np.random.multinomial(1, pred_prob))
    # to character
    pred_char = n_to_char[pred_char]
    return pred_char


def write_poem(seed, model,  n_to_char, char_to_n, max_seq=128, max_words=150):
    poem = seed
    # word count
    word_counter = len(re.findall(r'\w+', poem))
    # ends poem generator if max word is passed or poem end with final dot ($)
    while (word_counter < max_words) | (poem[-1]=='$') | (len(poem) < 600):    
        # Prediction next character
        next_char = predict_next_char(poem,
                                      n_to_char, char_to_n, model, max_seq)
        # append
        poem = poem + next_char
        # update word count
        word_counter = len(re.findall(r'\w+', poem))
    # add signature
    poem = poem + '\n\nEscrito por: AISP'
    return poem
        

# Generate poem
print("\n\n---- Creative moment - I'm writting a poem")
# random sentence to use as initial seed for the model
text, sequence = random_sentence(corpus, min_seq=90, max_seq=MAX_SEQ)
print('\nOriginal poem:\n\n', text)
print('\nSeed Sentence:\n\n', sequence)

poem = write_poem(sequence, model,  n_to_char, char_to_n,
                  max_seq=MAX_SEQ, max_words=60)

print('\nThis was an AI poem:\n\n', poem)



