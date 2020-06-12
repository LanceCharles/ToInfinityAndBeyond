import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout,GRU, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model,Sequential
from keras.callbacks import TensorBoard,EarlyStopping
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.optimizers import SGD
import tensorflow as tf
import pickle
path = 'NLP/'
tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
EMBEDDING_FILE=f'{path}word_embeddings.pkl'
TRAIN_DATA_FILE=f'{path}TIL_NLP_train_dataset.csv'
TEST_DATA_FILE=f'{path}TIL_NLP_test_dataset.csv'
embed_size = 100 # how big is each word vector
max_features = 5 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 20 # max number of words in a comment to use
callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5)
train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)
list_sentences_train = train["word_representation"].fillna("_na_").values
# print(len(list_sentences_train))
num = 0
for i in list_sentences_train:
    if len(i.split(" "))>num:
        num = len(i.split(" "))
print(num)

list_classes = ["outwear", "top", "trousers", "women dresses", "women skirts"]
y = train[list_classes].values
list_sentences_test = test["word_representation"].fillna("_na_").values

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
#
filePath = "NLP/word_embeddings.pkl"
infile = open(EMBEDDING_FILE,'rb')
newDict = pickle.load(infile)
#
all_embs = np.stack(newDict.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
print(emb_mean)
print(emb_std)

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = newDict.get(word)
    if embedding_vector is not None:
        print("===========================================")
        print(embedding_matrix[i])
        print("===========================================")
        print(embedding_vector)
        embedding_matrix[i] = embedding_vector


# inp = Input(shape=(maxlen,))
# x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
# x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
# x = GlobalMaxPool1D()(x)
# x = Dense(50, activation="relu")(x)
# x = Dropout(0.1)(x)
# # Sigmoid > Softmax as one synopsis may have many possible label
# x = Dense(5, activation="sigmoid")(x)
# model = Model(inputs=inp, outputs=x)
# model.summary()
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# #
# model.fit(X_t, y, batch_size=32, epochs=10, validation_split=0.2, callbacks=[callback]);

# model = Sequential()
# model.add(Embedding(max_features, embed_size, weights=[embedding_matrix]))
# model.add(Bidirectional(LSTM(50, return_sequences=True)))
# model.add(Dropout(0.5))
# model.add(Bidirectional(LSTM(64)))
# model.add(Dropout(0.5))
# model.add(Dense(5,activation="sigmoid"))
# model.summary()
#
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X_t, y, batch_size=32, epochs=10, validation_split=0.2, callbacks=[callback]);


deep_inputs = Input(shape=(maxlen,))
embedding_layer = Embedding(max_features, embed_size, weights=[embedding_matrix])(deep_inputs)
LSTM_Layer_1 = Bidirectional(LSTM(128))(embedding_layer)
dense_layer_1 = Dense(5, activation='sigmoid')(LSTM_Layer_1)
model = Model(inputs=deep_inputs, outputs=dense_layer_1)
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_t, y, batch_size=32, epochs=100, validation_split=0.2, callbacks=[callback]);


# emb_mean,emb_std
# infile.close()
# print(newDict)
# print(type(newDict))