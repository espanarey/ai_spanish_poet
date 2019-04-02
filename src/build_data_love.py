# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 22:41:47 2019

@author: reynaldo.espana.rey

Build data for spanish love poems
To be used during the transfer learning

source: https://www.analyticsvidhya.com/blog/2018/03/text-generation-using-python-nlp/
"""


# =============================================================================
# Libraries
# =============================================================================
import numpy as np
import pandas as pd
import re
import pickle
import nltk
import sys
sys.path.append('./src')
from utils import read_corpus, merge_corpus
from utils import vocab_mapping, corpus_split, build_data
# =============================================================================
# Input arguments
# =============================================================================
# is it a test? if True only 25 songs are used
TEST_MODE = False

# Where is the folder with all the corpus docs?
CORPUS_PATH = './data/DB/love poems/'
# Load same word mapping form all poems
DATA_POEMS_PATH = './data/data_proccessed/NLP_data_poems_120_no-split'

# Split train/test
# DO NOT SPLIT (few docs in corpus)
SPLIT = 1

# NLP
MAX_SEQ = 120 # sequence length
STRIDE = [MAX_SEQ/2, MAX_SEQ] 

# output path
OUTPUT_PATH = './data/data_proccessed/'
OUTPUT_FILE = 'NLP_data_love_120'


# =============================================================================
# Read data
# =============================================================================
# read all .txt poems files
corpus = read_corpus(CORPUS_PATH)

# If it is a test mode just take first 25 songs
if TEST_MODE:
    print("TEST MODE: ON")
    corpus = corpus[:250]
else:
    print("TEST MODE: OFF")

# =============================================================================
# Some Stats and Data Cleansing
# =============================================================================

############ Character per doc.
chars = [len(x['corpus']) for x in corpus]
print('\nCharacters per doc:\n', pd.Series(chars).describe())
# remove outliers, less than 260 characters
idx = [x > 260 for x in chars]
print('docs removed:', len(corpus) - sum(idx))
corpus = list(np.array(corpus)[np.array(idx)])


############ lines per doc, remove single liners
lines = [len(x['corpus'].split('\n')) for x in corpus]
print('\nLines per doc:\n', pd.Series(lines).describe())
# remove outliers. less than 10 lines
idx = [((x > 8) & (x < 50)) for x in lines]
print('docs removed:', len(corpus) - sum(idx))
corpus = list(np.array(corpus)[np.array(idx)])


############ words count.
words = [len(re.findall(r'\w+', x['corpus'])) for x in corpus]
print('\nwords per doc:\n', pd.Series(words).describe())
# remove outliers
# less than 100 words
idx = [((x > 30) & (x < 200)) for x in words]
print('docs removed:', len(corpus) - sum(idx))
corpus = list(np.array(corpus)[np.array(idx)])

words = [re.findall(r'\w+', x['corpus']) for x in corpus]
# flat list into single array to count frequency
words = [item for sublist in words for item in sublist]

print('\nTotal words in corpus', len(words))
print('\nUnique words:', len(set(words)))

# most common words couont
words_counter = nltk.FreqDist(words)
words_counter = pd.DataFrame(words_counter.most_common())
words_counter.columns = ['words', 'freq']
words_counter['len'] = [len(x) for x in words_counter['words']]
print('\Most common words:', '\nat least 4 digits:\n',
      words_counter[['words','freq']][words_counter['len']>=4].head(10),
      '\nat least 6 digits:\n',
      words_counter[['words','freq']][words_counter['len']>=6].head(10))
 

##### Clean less frecuent characters
# all text as single string
all_text = merge_corpus(corpus)
# unique characters
characters = sorted(list(set(all_text)))
print('unique characters:', len(characters))

# count of characters
print('Number of appereance per unique character in corpus')
chars_count = []
for digit in characters:
    counter = {'digit': digit,
                        'count': sum([digit in char for char in all_text])}
    print(counter)
    chars_count.append(counter)
chars_count = pd.DataFrame(chars_count)

# remove non meaningfull characters
# manually make the cut at 766
# chars_count['digit'][chars_count['count'] <= 766].values
chars_remove = ['\t', '"', '%', '&', "'", '\*', '\+', '/', '0', '1', '2', '3', '4',
                '5', '6', '7', '8', '9', '=', '\[', '\]', '_',  '`', '{',
                '¨', 'ª', '«', '²', '´', 'º', '»', 'à', 'â', 'ã', 'ä', 'ç',
                'è', 'ê', 'ì', 'ï', 'ò', 'ô', 'õ', 'ö', 'ü', '\xa0', '°']

print('\nErase from docs non meaningful characters:', chars_remove)
# as reggex expression
chars_remove = '[' + ''.join(chars_remove) + ']'
# erase from docs non meaninful characters
for doc in corpus:
    doc['corpus'] = re.sub(chars_remove, ' ', doc['corpus'])
    doc['corpus'] = re.sub(' +', ' ', doc['corpus']).strip()

# line space: '\r\n ' '\r\n' to '\n', '\r\r\n'
for doc in corpus:
    for pattern in ['\r\r\n', '\r\n ', '\r\n', '\n\n']:
        doc['corpus'] = re.sub(pattern, '\n', doc['corpus'])

# add special charater for at the begining and final of text.
# Model will learn when to start/end
for doc in corpus:
    doc['corpus'] = doc['corpus'] + '.$'


# all docs as single string
all_text = merge_corpus(corpus)
# unique characters
characters = sorted(list(set(all_text)))
print('unique characters after cleansing', len(characters))
print(''.join(characters))

# =============================================================================
# character/word mappings
# =============================================================================
# using same mappings of generic poems dataset
# word mapping generated with build_data_poems.py
# create dictionaries for character/number mapping
print('\n---\nLoading word mapping dictionaries')

# load data file
with open(DATA_POEMS_PATH + '.pickle', 'rb') as file:
    data_poems = pickle.load(file)
    
# Characters list    
characters = data_poems['words_mapping']['characters']
    
# dictionaries to be used as index mapping
n_to_char = data_poems['words_mapping']['n_to_char']
char_to_n = data_poems['words_mapping']['char_to_n']

# TODO: mapping used as part of the model
words_mapping = {'characters': characters,
                'n_to_char': n_to_char,
                'char_to_n': char_to_n}


# =============================================================================
# split train and test
# =============================================================================
# Train/Test per doc
corpus_train, corpus_test = corpus_split(corpus, split=SPLIT)

# =============================================================================
# Build Tensor Data
# =============================================================================
# Create tensor data from corpus
print('\n---\nBuild Tensor data')
# Train datasets
print('\nBuild Train data:')
train_x, train_y = build_data(corpus_train, char_to_n, 
                              max_seq = MAX_SEQ, stride=STRIDE)

if len(corpus_test):
    print('\nBuild Test data:')
    test_x, test_y = build_data(corpus_test, char_to_n, 
                                  max_seq = MAX_SEQ, stride=STRIDE)
else:
    test_x, test_y = None, None

# =============================================================================
# Save proccess data
# =============================================================================

# Merge all data in one dict
data = {'corpus': corpus,
        'words_mapping': words_mapping,
        'train_x': train_x ,
        'train_y': train_y,
        'test_x' : test_x,
        'test_y' : test_y}

# save file
with open(OUTPUT_PATH + OUTPUT_FILE + '.pickle', 'wb') as file:
    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
print('Data saved in:', OUTPUT_PATH + OUTPUT_FILE + '.pickle')
