#%% Import and function declaration
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Parameters
DATA_PATH = 'data/data_proccessed/npl_words__seq_len_5'
MODELS_PATH = 'models/'


def build_model(vocabulary_size: int, embedding_dim: int, rnn_units: int, rnn_layers: int, batch_size: int):
    """
    Build deep learning recurrent model
    :param vocabulary_size: number of different uniques words present in dataset
    :param embedding_dim: dimension of the embedding we want to train
    :param rnn_units: number of recurrent units used per layer
    :param rnn_layers: number of recurrent layers stacked
    :param batch_size: batch size used during training
    :return: Kera's model defined
    """

    if tf.test.is_gpu_available():
        rnn = tf.keras.layers.CuDDN
    else:
        import functools
        rnn = functools.partial(
            tf.keras.layers.GRU, recurrent_activation='sigmoid')

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocabulary_size, embedding_dim, batch_input_shape=[batch_size, None]))

    for layer in range(rnn_layers):
        model.add(tf.keras.layers.Bidirectional(
            rnn(rnn_units, return_sequences=True, recurrent_initializer='glorot_uniform', stateful=True)))

    model.add(tf.keras.layers.Dense(vocabulary_size))
    print(model.summary())

    return model


def find_word_key(dictionary: dict, word: str) -> int:
    """
    Given a mapper with keys as integers and values as strings, finds the key for a given word
    :param dictionary: dictionary to search in
    :param word: word to look for
    :return: numeric key position of the word; if it does not exist, returns None
    """
    try:
        key = list(dictionary.values()).index(word)
    except ValueError:
        key = None

    return key


def generate_text(model, map_id_word: dict, start_words: [str]):

    num_words_generate = 50

    # Converting our start string to numbers (vectorizing)
    input_eval = [find_word_key(map_id_word, x) for x in start_words]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_words_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the word returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(map_id_word[predicted_id])

    return " ".join(start_words) + " " + " ".join(text_generated)


#%% Model reconstruction
RNN_UNITS = 128
RNN_LAYERS = 2
EMBEDDING_DIM = 1024
VOCABULARY_SIZE = 52868

model = build_model(vocabulary_size=VOCABULARY_SIZE, embedding_dim=EMBEDDING_DIM, rnn_units=RNN_UNITS,
                    rnn_layers=RNN_LAYERS, batch_size=1)


model.load_weights(MODELS_PATH + 'model_04__epoch_14__loss_0.9475226238756985')
model.build(tf.TensorShape([1, None]))
model.summary()


#%% Predictor
with open(DATA_PATH + '.pickle', 'rb') as file:
    data = pickle.load(file)

map_n_to_word = data['words_mapping']
del data

poem_seed = ['el', 'amor', 'todo', 'lo', 'poder', 'y']


print(generate_text(model=model, map_id_word=map_n_to_word, start_words=poem_seed))