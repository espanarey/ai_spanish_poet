# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 14:29:10 2019

@author: reynaldo.espana.rey

Create Spanish love poems from a seed from user

%run ./src/write_poem.py
"""


# =============================================================================
# Libraries
# =============================================================================
import warnings
warnings.filterwarnings("ignore")
import pickle
import sys
sys.path.append('./src')
from utils import build_model, write_poem, valid_name


# =============================================================================
# Input arguments
# =============================================================================
# is it a test? if True only 25 songs are used
TEST_MODE = False

# Where is the folder with all the corpus docs?
CORPUS_PATH = './data/data_proccessed/NLP_data_poems_120_no-split'
# NLP
MAX_SEQ = 120

# Network params
MODEL_OUTPUT = './models/'
MODEL_NAME = 'seq-120_layers-5-512_encoding-128_batch-256_TL'
ENCODING_OUT = 128
HIDDEN_UNITS = [512, 512, 512, 512, 512]
MAX_WORDS = 200

# How creative do you want the model to be
CREATIVITY = 7 # 1: high creativity , 10: conservative
PATH_POEMS = 'poems_written/'

# =============================================================================
# Load Data
# =============================================================================
# load data file
with open(CORPUS_PATH + '.pickle', 'rb') as file:
    data = pickle.load(file)

# unpack elements needed for the model
words_mapping = data['words_mapping']
characters = words_mapping['characters']
n_to_char = words_mapping['n_to_char']
char_to_n = words_mapping['char_to_n']


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
print("\n\n---- Loading AI model\n\n")
model.load_weights(str(MODEL_OUTPUT + MODEL_NAME + '.h5'))

# Generate poem
while True:
    try:
        seed = input('How do you want to start the poem?\n\n')
        # create poem
        print("\n\n---- Creative moment - I'm thinking ...")
        print('\n\n\nWriting poem ... \n\n')
        poem = write_poem(seed, model, n_to_char, char_to_n,
                          max_seq = MAX_SEQ, max_words = MAX_WORDS, 
                          creativity=CREATIVITY)
        # write poem as .txt in putpus
        filename = valid_name(poem.split('\n')[0])
        with open(PATH_POEMS + filename + '.txt', "w") as text_file:
                    text_file.write(poem)
        # another poem?
        repeat = input('Do you want another poem? (Y/N) ')
        if repeat.lower() == 'n':
            print("Bye")
            break
    except KeyboardInterrupt:
        print("Bye")
        break
    
        
