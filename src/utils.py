# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:12:55 2019

Utils functions used in the project across different scripts
"""

# =============================================================================
# Libraries
# =============================================================================

import numpy as np
import re
import os
from sklearn.utils import shuffle 
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from tensorflow.keras.losses import sparse_categorical_crossentropy

# =============================================================================
# Read docs as corpus
# =============================================================================
# read song corpus
def read_corpus(path):
    # list of files
    files = os.listdir(path)
    print('\nReading:', len(files), 'files')
    # placeholder for results
    corpus = []
    # loop all files and read
    for file in files:
        with open(path + file, "rb") as text_file:
            doc = text_file.read()
            doc = doc.decode('latin-1')
            # lower case
            doc = doc.lower()
            # remove leading, ending and duplicates whitespaces
            doc = re.sub(' +', ' ', doc).strip()
            corpus.append({'file': path + file,
                           'corpus': doc})
    return corpus

# put together all the text to take unique characters
def merge_corpus(corpus):
    all_text = str()
    for x in corpus:
        all_text = all_text + x['corpus']
    print('Number of characters in corpus:', len(all_text))
    return all_text

# valid file name
def valid_name(value):
    value = re.sub('[^\w\s-]', '', value).strip().lower()
    value = re.sub('[-\s]+', '-', value)
    return value


# =============================================================================
# character/word mappings
# =============================================================================
def vocab_mapping(corpus):
    '''
    Mapping is a step in which we assign an arbitrary number to a character/word
    in the text. In this way, all unique characters/words are mapped to a number.
    This is important, because machines understand numbers far better than text,
     and this subsequently makes the training process easier.
    '''
    
    # create dictionaries for character/number mapping
    print('\n---\nCreating word mapping dictionaries')
    
    # all docs as single string
    all_text = merge_corpus(corpus)
    # unique characters
    characters = sorted(list(set(all_text)))
    print('unique characters after cleansing', len(characters))
    print(''.join(characters))
    
    # dictionaries to be used as index mapping
    n_to_char = {n:char for n, char in enumerate(characters)}
    char_to_n = {char:n for n, char in enumerate(characters)}
    
    return characters, n_to_char, char_to_n  


# =============================================================================
# split train and test
# =============================================================================
def corpus_split(corpus, split):
    # Train/Test per doc
    # create corpus index
    idx = [i for i in range(len(corpus))]
    # random random for train
    idx_train = np.random.choice(idx, size=int(len(corpus)*split), replace=False)
    # index not in train
    idx_test = [i for i in idx if i not in idx_train]
    # split corpus by index
    corpus_train = [corpus[i] for i in idx_train]
    corpus_test = [corpus[i] for i in idx_test]    
    # Docs stats corpus
    print('\n--- Total docs in corpus', len(corpus))
    print('split in:')
    print('Train:', len(corpus_train))
    print('Test:', len(corpus_test))
    return corpus_train, corpus_test

# =============================================================================
# Build Tensor Data
# =============================================================================

# Create tensor data from corpus
def build_data(corpus, char_to_n, max_seq = 100, stride = [1,6]):
    '''
    Transform list of documents into tensor format to be fed to a lstm network
    outout shape: (sequences, max_lenght)
    max_seq: maximum length of sequence
    stride: steps apply in rolling window over text. next windows could be next character(1) or 6 characters ahead
    '''
    # place holder to  save results
    data_x = []
    data_y = []
    sequences = []
    # target sequence is lagged by 1, hence sequence length = max_seq+1
    max_seq+=1  
    # for each document in corpus
    for i in range(len(corpus)):
        if (i % max(1, int(len(corpus)/10)) == 0):
            print('\n--- Progress %:{0:.2f}'.format(i/len(corpus)))
        text = corpus[i]['corpus']
        text_length = len(text)
        # iterate for all text in rolling windows of size max_seq
        j = max_seq
        while j < text_length + stride[0]:
            k_to = min(j, text_length) # 
            k_from = (k_to - max_seq)            
            #print(j, ':', k_from, '-', k_to)
            #♠ slice text
            sequence = text[k_from:k_to] 
            # characters to int
            sequence_encoded = np.array([char_to_n[x] for x in sequence]) 
            # append results
            sequences.append(sequence)
            data_x.append(sequence_encoded[:-1])
            data_y.append(sequence_encoded[1:])
            # random stride between 1-6
            j+=int(np.random.uniform(stride[0], stride[1]+1))       
    # Tensor structure
    data_x = np.array(data_x) 
    data_y = np.array(data_y) 
    
    # Shuffle data
    data_x, data_y = shuffle(data_x, data_y)
    # output
    print('Outupt shape -', 'X:', data_x.shape, '- Y:', data_y.shape)
    size = data_x.nbytes*1e-6 + data_y.nbytes*1e-6
    size = print(int(size), 'Megabytes')
    return data_x, data_y


# =============================================================================
# LSTM Model
# =============================================================================

# Model loss
def loss(y_true, y_pred):
    return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

# model params
def build_model(batch_size, encoding_dim, hidden_units, optimizer, loss=loss):
    '''
    encoding_dim [in, out]
    '''
    print('\n---\nModel network')
    # build model
    model = Sequential()
    # embeding
    model.add(Embedding(input_dim = encoding_dim[0], output_dim = encoding_dim[1],
                        batch_input_shape = [batch_size, None]))
    # LSTM Layers
    for units in hidden_units:
        model.add(LSTM(units, return_sequences=True, recurrent_initializer='glorot_uniform',
                   dropout=0.25, recurrent_dropout=0.05, consume_less='mem'))
    # Final layer
    model.add(Dense(encoding_dim[0]))
    
    # compile
    model.compile(loss = loss,
                  optimizer = optimizer)  # RMSprop, adam
    # architecture    
    print(model.summary())
    return model



# =============================================================================
# Model inference
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
def predict_next_char(sequence, n_to_char, char_to_n, model, max_seq=128, creativity=3):
    '''
    sequence: input sequence seen so far. (if blank model will start with a random character)
    n_to_char, char_to_n: Vocabulary dictionaries used in training
    model: trained model weights
    max_seq: maximum number of characters to seen in one sequence (use the same sequence as model)
    creativity: 1: super creative, 10: Conservative.
    '''
    # start with a random character
    if len(sequence)==0:
        sequence = '\n'
    # cut sequence into max length allowed   
    sequence = sequence[max(0, len(sequence)-max_seq):].lower() 
    # transform sentence to numeric
    sequence_encoded = np.array([char_to_n[char] for char in sequence])
    # reshape for single batch predicton
    sequence_encoded = np.reshape(sequence_encoded, (1, len(sequence_encoded)))
    # model prediction
    pred_encoded = model.predict(sequence_encoded)  
    # last prediction in sequence
    pred = pred_encoded[0][-1]
    # from log probabilities to normalized probabilities
    pred_prob = np.exp(pred)
    pred_prob = np.exp(pred)/(np.sum(np.exp(pred))*1.0001)
    # get index of character  based on probabilities
    # add an extra digit (issue from np.random.multinomial)
    pred_char = np.random.multinomial(creativity, np.append(pred_prob, .0))
    # character with highest aperances
    chars_max = pred_char==pred_char.max()
    # get index of those characters
    chars_max_idx = [i for i in range(len(pred_char)) if chars_max[i]]
    char_idx = np.random.choice(chars_max_idx, 1)[0]
    # if prediction do not match vocabulary. do nothing
    if char_idx > len(n_to_char)-1: char_idx = ''
    char = n_to_char[char_idx]
    return char


def write_poem(seed, model,  n_to_char, char_to_n, max_seq=128, max_words=150, creativity=3):
    # start poem with the seed
    poem = seed
    print(poem, end ="")
    # placeholder stopers 
    final = True
    word_counter = len(re.findall(r'\w+', poem)) < max_words
    # ends poem generator if max word is passed or poem end with final dot ($)
    while (word_counter & final):    
        # Prediction next character
        next_char = predict_next_char(poem, n_to_char, char_to_n, 
                                      model, max_seq, creativity)
        print(next_char, end ="")
        # append
        poem = poem + next_char
        # update stopers
        final = poem[-1] != '$'
        word_counter = len(re.findall(r'\w+', poem)) < max_words        
    # add signature
    signature = '\n\nAI.S.P\n\n'
    print(signature, end ="")
    poem = poem + signature
    return poem
        