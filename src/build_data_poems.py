#%% Import and function declaration
import numpy as np
import os
import re
import pickle

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
seq_length = 5
output_file = 'npl_words__seq_len_'+str(seq_length)


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
    text = re.sub(r'/[^\S\r\n]/', ' ', text).strip()  # spacing characters except '\n'
    text = re.sub(' +', ' ', text)
    text = re.sub('[^a-z \n]', '', text)

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


def process_corpus(path: str, corpus_size: int = 100000) -> list:
    """
    Read the all text files present in the path and applies text cleansing
    :param path: path where to look for text files
    :param corpus_size: corpus size, required to limit the dataset
    :return: corpus of data
    """
    files = os.listdir(path)[0:corpus_size]
    corpus = []

    for file in files:
        with open(path + file, "rb") as text_file:
            document = text_file.read()
            document = document.decode('latin-1')
            document = document.lower()
            document = remove_accented_chars(text=document)
            document = remove_special_characters(text=document)
            document = lemmatize_text(text=document, lemmatizer=nlp)
            # document = remove_stopwords(text=document, tokenizer=tokenizer)
            document = document.split(' ')
            document = [word for word in document if word != '']
            corpus.append({'file': path + file, 'text': document})

    return corpus


def create_vocabulary(text_corpus: list) -> list:
    """
    Creates vocabulary of unique words present in the corpus
    :param text_corpus: corpus to use as vocabulary base
    :return: vocabulary
    """
    all_words = list()
    for element in text_corpus:
        all_words.append(element['text'])
    all_words = [item for sublist in all_words for item in sublist]

    return sorted(list((set(all_words))))


def map_corpus(text_corpus: list, words_mapper: dict) -> list:
    """
    Maps a corpus to a given words mapper
    :param text_corpus: corpus of text to be mapped
    :param words_mapper: mapper used to translate words into integers
    :return: corpus mapped
    """
    num_corpus = []
    for element in text_corpus:
        num_corpus.append([words_mapper[word] for word in element['text']])

    return num_corpus


def create_dataset(numerical_corpus: list, sequence_length: int, target_words_stride: int = 1) -> list:
    """
    Given a numerical corpus, it generates the dataset for training
    :param numerical_corpus: corpus already mapped into integers
    :param sequence_length: sequence length being fed into the algorithm
    :param target_words_stride: number of words we displace between the train data and the target
    :return: dataset properly formated for training
    """
    dataset = []

    for document in numerical_corpus:
        document_data = []
        num_splits = (len(document)-target_words_stride) // sequence_length

        for i_step in range(num_splits):
            if i_step == 0:
                i_position = 0
            else:
                i_position += sequence_length
            document_data.append([
                document[i_position:(i_position + sequence_length)],
                document[(i_position + target_words_stride):(i_position + sequence_length + target_words_stride)]])

        dataset.append(document_data)

    return [item for sublist in dataset for item in sublist]


def split_train_test(dataset: list, train_ratio: float = 0.9) -> tuple:
    """
    Splits the dataset into train/test split based on the selected ratio
    :param dataset: dataset already processed in to neural net format
    :param train_ratio: ratio of the total data used for training
    :return: train_data and test_data
    """
    idx = [i for i in range(len(dataset))]
    idx_train = np.random.choice(idx, size=int(len(dataset) * train_ratio), replace=False)
    idx_test = [i for i in idx if i not in idx_train]
    dataset_train = [dataset[i] for i in idx_train]
    dataset_test = [dataset[i] for i in idx_test]

    print('Corpus Docs:', len(corpus), '\nTrain data:', len(dataset_train), '- Test data:', len(dataset_test))
    return dataset_train, dataset_test


#%% Main program
corpus = process_corpus(corpus_path)
vocabulary = create_vocabulary(corpus)

map_n_to_word = {n: word for n, word in enumerate(vocabulary)}
map_word_to_n = {word: n for n, word in enumerate(vocabulary)}
corpus_mapped = map_corpus(text_corpus=corpus, words_mapper=map_word_to_n)

dataset = create_dataset(numerical_corpus=corpus_mapped, sequence_length=seq_length)
train_data, test_data = split_train_test(dataset=dataset, train_ratio=0.8)

data = {'corpus': corpus, 'words_mapping': map_n_to_word, 'train_data': train_data , 'test_data' : test_data}

with open(output_path + output_file + '.pickle', 'wb') as file:
    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
print('Data saved in:', output_path + output_file + '.pickle')
