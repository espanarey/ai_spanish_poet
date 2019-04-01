# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 22:41:47 2019

@author: reynaldo.espana.rey

Build data for song corpus

source: https://www.analyticsvidhya.com/blog/2018/03/text-generation-using-python-nlp/
"""


# =============================================================================
# Libraries
# =============================================================================
import numpy as np
import pandas as pd
import os
import re
import pickle
from sklearn.utils import shuffle 
from keras.utils import np_utils
import nltk


# =============================================================================
# Input arguments
# =============================================================================
# is it a test? if True only 25 songs are used
TEST_MODE = False

# Where is the folder with all the corpus docs?
CORPUS_PATH = '../data/DB/spanish poems/'

# Split train/test
SPLIT = .9

# NLP
MAX_SEQ = 100
MIN_SEQ = 50
STRIDE = [5, 15]

# output path
OUTPUT_PATH = '../data/data_proccessed/'
OUTPUT_FILE = 'NLP_data_poems_100-50'




# =============================================================================
# Read data
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

# read all .txt lyrics files
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
# unique characters count. character unique songs
# put together all the text to take unique characters
def merge_corpus(corpus):
    all_text = str()
    for x in corpus:
        all_text = all_text + x['corpus']
    print('Number of characters in corpus:', len(all_text))
    return all_text

# all songs as single string
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


# add special charater for final of text. Model will learn when to end
for doc in corpus:
    doc['corpus'] = doc['corpus'] + '.$'


# TODO: clean key words


# =============================================================================
# character/word mappings
# =============================================================================
'''
Mapping is a step in which we assign an arbitrary number to a character/word
in the text. In this way, all unique characters/words are mapped to a number.
This is important, because machines understand numbers far better than text,
 and this subsequently makes the training process easier.
'''

# create dictionaries for character/number mapping
print('\n---\nCreating word mapping dictionaries')

# all songs as single string
all_text = merge_corpus(corpus)
# unique characters
characters = sorted(list(set(all_text)))
print('unique characters after cleansing', len(characters))
print(''.join(characters))

# dictionaries to be used as index mapping
n_to_char = {n:char for n, char in enumerate(characters)}
char_to_n = {char:n for n, char in enumerate(characters)}

# TODO: mapping used as part of the model
words_mapping = {'characters': characters,
                'n_to_char': n_to_char,
                'char_to_n': char_to_n}




# =============================================================================
# split train and test
# =============================================================================
# Train/Test per doc
# create corpus index
idx = [i for i in range(len(corpus))]
# random random for train
idx_train = np.random.choice(idx, size=int(len(corpus)*SPLIT), replace=False)
# index not in train
idx_test = [i for i in idx if i not in idx_train]
# split corpus by index
corpus_train = [corpus[i] for i in idx_train]
corpus_test = [corpus[i] for i in idx_test]

# Docs stats corpus
print('\n--- Total docs in corpus', len(corpus))
print('split in:', 'Train:', len(corpus_train), '- Test:', len(corpus_test))



# =============================================================================
# Build Tensor Data
# =============================================================================

# Create tensor data from corpus
def build_data(corpus, normalized_by=None, max_seq = 128, min_seq = 64, stride = [1,6], nans=0):
    '''
    Transform list of documents into tensor format to be fed to a lstm network
    outout shape: (sequences, max_lenght, 1)
    max_seq: maximum length of sequence
    min_seq: minimun character per sequence, rest is fill with NAs
    stride: steps apply in rolling window over text. next windows could be next character(1) or 6 characters ahead
    normalized_by: if not None, values are normalized by the number of unique characters in the corpus
    '''
    # place holder to  save results
    data_x = []
    data_y = []
    sequences = []
    # for each document in corpus
    for i in range(len(corpus)):
        if (i % max(1, int(len(corpus)/10)) == 0):
            print('\n--- Progress %:{0:.2f}'.format(i/len(corpus)))
        text = corpus[i]['corpus']
        text_length = len(text)
        # start with a index before 0 else first characters will appear only once
        j = 0
        while j < (text_length - min_seq):
            # Sequence lenght between max and min
            seq_length = int(np.random.normal(max_seq*1.2, max_seq*.2))
            seq_length = min(max_seq, max(min_seq, seq_length))
            # get text sequence
            # print('from:', j, 'to:', min(text_length, j + seq_length))
            sequence = text[j:min(text_length-1, j + seq_length)]
            # do not cut the first word. always start with a full word
            # when the first word starts (look the first space or)
            first_word = np.argmax([x==' ' for x in sequence])
            # cut sequence
            sequence = sequence[first_word+1:]
            if len(sequence) >= min_seq:       
                sequence_encoded = [char_to_n[char] for char in sequence]
                sequence_encoded = np.reshape(sequence_encoded, (len(sequence),1))
                # use sequence placeholder to fill in nans
                sequence_x = np.full((max_seq, 1), np.nan, dtype=np.float32)
                sequence_x[len(sequence_x)-len(sequence_encoded):] = sequence_encoded
                # next character as target variable
                label = text[min(text_length-1, j + seq_length)]
                label_decoded = char_to_n[label]
                # append results
                sequences.append(sequence)
                data_x.append(sequence_x)
                data_y.append(label_decoded)
            # random stride between 1-6
            j+=int(np.random.uniform(stride[0], stride[1]+1))
    # Tensor structure
    data_x = np.reshape(data_x, (len(data_x), max_seq, 1))
    # Normalized data
    if normalized_by is not None:
        data_x = (data_x + 1) / float(normalized_by) # add delta so nans will be zero
    # replace nans by -1
    data_x[np.isnan(data_x)] = nans
    # target variable to categorical dummy
    data_y = np_utils.to_categorical(data_y)
    # Shuffle data:
    data_x, data_y, sequences = shuffle(data_x, data_y, sequences)
    # output
    print('Outupt shape -', 'X:', data_x.shape, '- Y:', data_y.shape)
    size = data_x.nbytes*1e-6 + data_y.nbytes*1e-6
    size = print(int(size), 'Megabytes')
    return data_x, data_y

# Train datasets
print('\n---\nBuild Tensor data')

print('\nBuild Train data:')
train_x, train_y = build_data(corpus_train,
                              normalized_by=len(characters),
                              max_seq = MAX_SEQ, min_seq = MIN_SEQ, stride=STRIDE)

print('\nBuild Test data:')
test_x, test_y = build_data(corpus_test,
                            normalized_by=len(characters),
                            max_seq = MAX_SEQ, min_seq = MIN_SEQ, stride=STRIDE)



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
