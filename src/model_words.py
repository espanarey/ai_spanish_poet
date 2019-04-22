#%% Import and function declaration
from __future__ import absolute_import, division, print_function

import pickle
import numpy as np
import tensorflow as tf

# Parameters
data_path = 'data/data_proccessed/npl_words__seq_len_5'
models_path = 'models/'


def build_model(vocabulary_size: int, embedding_dim: int, rnn_units:int, rnn_layers:int, batch_size:int):
    """
    Build deep learning recurrent model
    :param vocabulary_size: number of different uniques words present in dataset
    :param embedding_dim: dimension of the embedding we want to train
    :param rnn_units: number of recurrent units used per layer
    :param rnn_layers: number of recurrent layers stacked
    :param batch_size: batch size used during training
    :return: Kera's model defined
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocabulary_size, embedding_dim, batch_input_shape=[batch_size, None]))

    for layer in range(rnn_layers):
        model.add(tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=False,
                                       recurrent_initializer='glorot_uniform'))

    model.add(tf.keras.layers.Dense(vocabulary_size))
    print(model.summary())

    return model


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


#%% Dataset load
with open(data_path + '.pickle', 'rb') as file:
    data = pickle.load(file)

print('unpack:', data.keys())
for item in data.keys(): exec(item + " = eval('data[item]') ")

del data

X_train = np.array([train_data[element][0] for element in range(len(train_data))])
y_train = np.array([train_data[element][1] for element in range(len(train_data))])
X_test = np.array([test_data[element][0] for element in range(len(test_data))])
y_test = np.array([test_data[element][1] for element in range(len(test_data))])

print('The shapes are: \n X_train: {a} | y_train: {b} \n X_test: {c}, y_test: {d}'.format(
    a = X_train.shape,b = y_train.shape, c = X_test.shape, d = y_test.shape))


#%% Model Definition
epochs = 30
batch_size = 512
rnn_units = 512
rnn_layers = 1

vocabulary_size = len(words_mapping)  # 52864
embedding_dim = 128

model = build_model(vocabulary_size=vocabulary_size, embedding_dim=embedding_dim, rnn_units=rnn_units,
                    rnn_layers=rnn_layers, batch_size=batch_size)

model.compile(optimizer='adam', loss=loss)

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=models_path + "model_01__" +
                                                       "epoch_{epoch}__val_loss_{val_loss}", save_weights_only=False)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.001)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.05, patience=3, verbose=1)

#%% Model training
history = model.fit(x=X_train,
                    y=y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.15,
                    callbacks=[checkpoint, reduce_lr, early_stopping])
