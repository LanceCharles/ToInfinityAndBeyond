import sys, os, re, csv, codecs, numpy as np, pandas as pd
from numpy.random import RandomState
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout,GRU, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D,Conv1D, MaxPooling1D, Embedding,GlobalMaxPooling1D,GlobalAveragePooling1D
from keras.models import Model,Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from keras.callbacks import TensorBoard,EarlyStopping
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.initializers import Constant
import tensorflow as tf
import pickle
path = 'NLP/'
EMBEDDING_FILE=f'{path}word_embeddings.pkl'
TRAIN_DATA_FILE=f'{path}TIL_NLP_train_dataset.csv'
TEST_DATA_FILE=f'{path}TIL_NLP_test_dataset.csv'
embed_size = 100 # how big is each word vector
max_features = 100 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 35 # max number of words in a comment to use
callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5)
train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)
list_sentences_train = train["word_representation"].values
list_sentences_test = test["word_representation"].fillna("_na_").values
list_classes = ["outwear", "top", "trousers", "women dresses", "women skirts"]
y = train[list_classes].values

print('Indexing word vectors')

emb_index= {}
filePath = "NLP/word_embeddings.pkl"
infile = open(EMBEDDING_FILE,'rb')
newDict = pickle.load(infile)
print('Found %s word vectors.' % len(newDict))

print('Processing text dataset')
inte = 0
uniquecombi = []
for array in list_sentences_train:
    for ind in array.split():
        if ind in uniquecombi:
            continue
        else:
            uniquecombi.append(ind)
print(len(uniquecombi))
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
word_index= tokenizer.word_index
print(word_index)
print("Found %s unique tokens." % len(word_index))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
print(list_tokenized_train)
data = pad_sequences(list_tokenized_train,maxlen=maxlen)
labels = (np.asarray(y))
print('Shape of data tensor:',data.shape)
print('Shape of label tensor:',labels.shape)

# indices = np.arange(data.shape[0])
# np.random.shuffle(indices)
# data = data[indices]
# labels = labels[indices]
# num_validation_samples = int(0.2 * data.shape[0])
# x_train = data[:-num_validation_samples]
# y_train = labels[:-num_validation_samples]
# x_val = data[-num_validation_samples]
# y_val = data[-num_validation_samples]

print('Preparing embedding matrix.')
print(min(max_features,len(word_index)+1))
num_words = min(max_features,len(word_index)+1)
print(len(word_index))
# embedding_matrix = np.zeros((num_words,embed_size))

all_embs = np.stack(newDict.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
print(emb_mean,emb_std)
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word,i in word_index.items():
    if i >= max_features: continue
    embedding_vector = newDict.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
embedding_layer = Embedding(num_words,embed_size,weights=[embedding_matrix])

print('Training model.')
sequence_input = Input(shape=(maxlen,))
embedded_sequences = embedding_layer(sequence_input)
LSTM_Layer_1 = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(embedded_sequences)
LSTM_Layer_1 = GlobalMaxPool1D()(LSTM_Layer_1)
LSTM_Layer_1 = Dense(50, activation="relu")(LSTM_Layer_1)
LSTM_Layer_1 = Dropout(0.1)(LSTM_Layer_1)
preds = Dense(5, activation='sigmoid')(LSTM_Layer_1)
model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())
model.fit(data,labels,batch_size=32,epochs=30,validation_split=0.2,callbacks=[callback]) #100 to 30

list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
y_test = model.predict([X_te], batch_size=32, verbose=1)
sample_submission = pd.read_csv(f'{path}NLP_submission_example.csv')
sample_submission[list_classes] = y_test
sample_submission.to_csv('submission.csv', index=False)


















# tokenizer = Tokenizer(num_words=max_features)
# tokenizer.fit_on_texts(list(list_sentences_train))
# list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
# list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
# X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
# X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
# #
# filePath = "NLP/word_embeddings.pkl"
# infile = open(EMBEDDING_FILE,'rb')
# newDict = pickle.load(infile)
# #
# all_embs = np.stack(newDict.values())
# emb_mean,emb_std = all_embs.mean(), all_embs.std()
# print(emb_mean)
# print(emb_std)
#
# sentences = train["word_representation"].values
# label = train["outwear", "top", "trousers", "women dresses", "women skirts"].values
# vectorizer = CountVectorizer(min_df=0,lowercase=False)
# vectorizer.fit(sentences)
# print(vectorizer.vocabulary_)
# vectorizer.transform(sentences).toarray()
# X_train,X_Test,Y_Train,Y_Test = train_test_split(sentences,label,test_size=0.2,random_state=42)
#
# vectorizer = CountVectorizer()
# vectorizer.fit(X_train)
#
# X_train_V = vectorizer.transform(X_train)
# X_test_V = vectorizer.transform(X_Test)
# classifier = LogisticRegression()
# classifier.fit(X_train_V,Y_Train)
# score = classifier.score(X_test_V,Y_Test)
# print(score)
#
#
# word_index = tokenizer.word_index
# nb_words = min(max_features, len(word_index))
# embedding_matrix = np.zeros((len(word_index) + 1, embed_size))#np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
# for word, i in word_index.items():
#     if i >= max_features: continue
#     embedding_vector = newDict.get(word)
#     if embedding_vector is not None:
#         embedding_matrix[i] = embedding_vector
#
#
# inp = Input(shape=(maxlen,))
# print(len(word_index))
# x = Embedding(len(word_index) + 1,embed_size, weights=[embedding_matrix],input_length=maxlen,trainable=False)(inp)
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
