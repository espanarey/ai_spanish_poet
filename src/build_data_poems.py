#%% Imports and function declaration
import numpy as np
import os
import re
import pickle
from sklearn.utils import shuffle

import nltk
import spacy
import unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('spanish')
nlp = spacy.load('es', parse=False, tag=False, entity=False)

# Parameters
corpus_path = 'data/DB/spanish poems/'
output_path = 'data/data_proccessed/'
output_file = 'npl_words__seq_len_5'


def remove_accented_chars(text: str) -> str:
    """
    Cleans accented characters from text
    :param text: text to process
    :return: text processed
    """
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    return text


def remove_special_characters(text: str) -> str:
    """
    Removes special characters
    :param text: text to process
    :return: text processed
    """
    for i_pattern in ['\r\r\n', '\r\n ', '\r\n', '\n\n']:  # line space: '\r\n ' '\r\n' to '\n', '\r\r\n'
        text = re.sub(i_pattern, '\n', text)
    text = re.sub(' +', ' ', text).strip()
    text = re.sub('[^a-z\n ]', '', text)

    return text


def lemmatize_text(text: str, lemmatizer: spacy.lang) -> str:
    """
    Group together the inflected forms of a word
    :param text: text to process
    :param lemmatizer: Spacy's lemmatizer
    :return: text processed
    """
    text = lemmatizer(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text


def remove_stopwords(text: str, tokenizer: nltk.tokenize.toktok.ToktokTokenizer) -> str:
    """
    Remove stopwords from text
    :param text: text to process
    :param tokenizer: nltk tokenizer
    :return: text processed
    """
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)

    return filtered_text


def process_corpus(path: str) -> list:
    """
    Read the all text files present in the path and applies text cleansing
    :param path: path where to look for text files
    :return: corpus of data
    """
    files = os.listdir(path)[1:10]
    corpus = []

    for file in files:
        with open(path + file, "rb") as text_file:
            document = text_file.read()
            document = document.decode('latin-1')
            document = document.lower()
            document = re.sub(' +', ' ', document).strip()
            document = remove_accented_chars(text=document)
            document = remove_special_characters(text=document)
            document = lemmatize_text(text=document, lemmatizer=nlp)
            # document = remove_stopwords(text=document, tokenizer=tokenizer)
            document = document.split()
            corpus.append({'file': path + file, 'text': document})

    return corpus


#%% Data load and pre-processing
corpus = process_corpus(corpus_path)



#%%
def merge_corpus(corpus):
    all_text = str()
    for x in corpus:
        all_text = all_text + x['text']
    print('Number of characters in corpus:', len(all_text))
    return all_text

# all songs as single string
all_text = merge_corpus(corpus)
# unique characters
characters = sorted(list(set(all_text)))
print('unique characters:', len(characters))


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
idx_train = np.random.choice(idx, size=int(len(corpus) * train_test_split), replace=False)
# index not in train
idx_test = [i for i in idx if i not in idx_train]
# split corpus by index
corpus_train = [corpus[i] for i in idx_train]
corpus_test = [corpus[i] for i in idx_test]

print('split corpus docs:', len(corpus),
      '\ntrain:', len(corpus_train), '- test:', len(corpus_test))



# =============================================================================
# Build Tensor Data
# =============================================================================
# TODO: do not cut the first word, do not start with blank space

# Create tensor data from corpus
def build_data(corpus, normalized_by=None, max_seq = 128, min_seq = 64, stride = [1,6], nans=-1.):
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
    # for each document in corpus
    for i in range(len(corpus)):
        if (i % int(len(corpus)/10) == 0):
            print('\n--- Progress %:{0:.2f}'.format(i/len(corpus)))
        text = corpus[i]['corpus']
        text_length = len(text)
        # start with a index before 0 else first characters will appear only once
        j = 0
        while j < (text_length - min_seq):
            # Sequence lenght between max and min
            seq_length = int(np.random.normal((max_seq + min_seq)/2, (max_seq+min_seq)/6))
            seq_length = min(max_seq, max(min_seq, seq_length))
            # get text sequence
            # print('from:', j, 'to:', min(text_length, j + seq_length))
            sequence = text[j:min(text_length-1, j + seq_length)]
            sequence_encoded = [char_to_n[char] for char in sequence]
            sequence_encoded = np.reshape(sequence_encoded, (len(sequence),1))
            # use sequence placeholder to fill in nans
            sequence_x = np.full((max_seq, 1), np.nan, dtype=np.float32)
            sequence_x[len(sequence_x)-len(sequence_encoded):] = sequence_encoded
            # next character as target variable
            label = text[min(text_length-1, j + seq_length)]
            label_decoded = char_to_n[label]
            # append results
            data_x.append(sequence_x)
            data_y.append(label_decoded)
            # random stride between 1-6
            j+=int(np.random.uniform(stride[0], stride[1]+1))
    # Tensor structure
    data_x = np.reshape(data_x, (len(data_x), max_seq, 1))
    # Normalized data
    if normalized_by is not None:
        data_x = data_x / float(normalized_by)
    # replace nans by -1
    data_x[np.isnan(data_x)] = nans
    # target variable to categorical dummy
    data_y = np_utils.to_categorical(data_y)
    # Shuffle data:
    data_x, data_y = shuffle(data_x, data_y)
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
with open(output_path + OUTPUT_FILE + '.pickle', 'wb') as file:
    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
print('Data saved in:', output_path + OUTPUT_FILE + '.pickle')
