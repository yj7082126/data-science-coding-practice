# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 21:55:40 2019

@author: user
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gzip
import spacy
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, TimeDistributed, Bidirectional, GlobalMaxPool1D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF('reviews_Beauty_5.json.gz')

df.rename(columns={'reviewerID': 'user_id', 
                        'asin': 'item_id', 
                        'reviewerName': 'user_name', 
                        'reviewText': 'review_text',
                        'summary': 'review_summary',
                        'overall': 'score'},
               inplace=True)

df.user_id = df.user_id.astype('category').cat.codes.values
df.item_id = df.item_id.astype('category').cat.codes.values
# Add IDs for embeddings.
df['user_emb_id'] = df['user_id']
df['item_emb_id'] = df['item_id']

df = df.sample(frac=0.1, random_state = 1)

#%%
nlp = spacy.load('en_core_web_md')
nlp.add_pipe(nlp.create_pipe('sentencizer'))
embeddings = nlp.vocab.vectors.data

#%%
from progressbar import ProgressBar
pbar = ProgressBar()

X = np.asarray(df['review_summary'])
Y = np.asarray(df['score'] >= 4.0)

#%%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train, test_size=(1/9), random_state=42)

X_train = list(nlp.pipe(X_train))
X_train = [[token.vocab.vectors.find(key=token.orth) for token in doc] for doc in X_train]
length = len(sorted(X_train ,key=len, reverse=True)[0])
X_train = np.array([xi+[None]*(length-len(xi)) for xi in X_train])

X_dev = list(nlp.pipe(X_dev))
X_dev = [[token.vocab.vectors.find(key=token.orth) for token in doc] for doc in X_dev]
X_dev = np.array([xi+[None]*(length-len(xi)) for xi in X_dev])

X_orig_test = X_test
X_test = list(nlp.pipe(X_test))
X_test = [[token.vocab.vectors.find(key=token.orth) for token in doc] for doc in X_test]
X_test = np.array([xi+[None]*(length-len(xi)) for xi in X_test])

Y_train = Y_train.astype(int).reshape(-1)
Y_dev = Y_dev.astype(int).reshape(-1)
Y_test = Y_test.astype(int).reshape(-1)

#%%
model = Sequential()
model.add(Embedding(embeddings.shape[0], embeddings.shape[1], 
                    input_length = 30, trainable=False, 
                    weights = [embeddings], mask_zero = True))
model.add(TimeDistributed(Dense(64, use_bias=False)))
model.add(Bidirectional(LSTM(64, recurrent_dropout=0.2, dropout=0.2)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#%%
history = model.fit(X_train, Y_train, epochs = 10, batch_size = 160, validation_data = (X_dev, Y_dev), shuffle=True)

loss, acc = model.evaluate(X_test, Y_test)
print("Test accuracy = ", acc)

#%%
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
x = range(1, len(acc)+1)

fig, ax = plt.subplots(2,1, figsize=(8,8))
ax[0].plot(x, acc, 'b', label='Training acc')
ax[0].plot(x, val_acc, 'r', label='Validation acc')
ax[0].set_title('Training and validation accuracy')
ax[0].legend()
ax[1].plot(x, loss, 'b', label='Training loss')
ax[1].plot(x, val_loss, 'r', label='Validation loss')
ax[1].set_title('Training and validation loss')
ax[1].legend()

plt.show()

#%%
pred = model.predict(X_test)
pred = np.asarray([1 if x[0] >= 0.5 else 0 for x in pred])

conf_mat = confusion_matrix(Y_test, pred)
f1 = f1_score(Y_test, pred)
#fn = []
#fp = []
#for i in range(len(X_test)):
#    if ((Y_test[i] == 1) & (pred[i] < 0.5)):
#        fn.append(X_orig_test[i])
#        print (str(i) + ": Predicted as Negative: " + X_orig_test[i])
#    elif ((Y_test[i] == 0) & (pred[i] >= 0.5)):
#        fp.append(X_orig_test[i])
#        print (str(i) + ": Predicted as Positive: " + X_orig_test[i])
        
#%%
model2 = Sequential()
model2.add(Embedding(embeddings.shape[0], embeddings.shape[1], 
                    input_length = 30, trainable=False, 
                    weights = [embeddings]))
model2.add(GlobalMaxPool1D())
model2.add(Dense(10, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model2.summary()       

#%%
history2 = model2.fit(X_train, Y_train, epochs = 40, batch_size = 40, validation_data = (X_dev, Y_dev), shuffle=True)

loss, acc = model2.evaluate(X_test, Y_test)
print("Test accuracy = ", acc)

#%%
acc = history2.history['acc']
val_acc = history2.history['val_acc']
loss = history2.history['loss']
val_loss = history2.history['val_loss']
x = range(1, len(acc)+1)

fig, ax = plt.subplots(2,1, figsize=(8,8))
ax[0].plot(x, acc, 'b', label='Training acc')
ax[0].plot(x, val_acc, 'r', label='Validation acc')
ax[0].set_title('Training and validation accuracy')
ax[0].legend()
ax[1].plot(x, loss, 'b', label='Training loss')
ax[1].plot(x, val_loss, 'r', label='Validation loss')
ax[1].set_title('Training and validation loss')
ax[1].legend()

plt.show()

#%%
model3 = Sequential()
model3.add(Embedding(embeddings.shape[0], embeddings.shape[1], 
                    input_length = 30, trainable=False, 
                    weights = [embeddings], mask_zero = True))
model3.add(LSTM(128, return_sequences=True))
model3.add(Dropout(0.5))
model3.add(LSTM(128, return_sequences=False))
model3.add(Dropout(0.5))
model3.add(Dense(1, activation='sigmoid'))
model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#%%
history3 = model3.fit(X_train, Y_train, epochs = 10, batch_size = 80, validation_data = (X_dev, Y_dev), shuffle=True)

loss3, acc3 = model3.evaluate(X_test, Y_test)
print("Test accuracy = ", acc)

#%%
acc = history3.history['acc']
val_acc = history3.history['val_acc']
loss = history3.history['loss']
val_loss = history3.history['val_loss']
x = range(1, len(acc)+1)

fig, ax = plt.subplots(2,1, figsize=(8,8))
ax[0].plot(x, acc, 'b', label='Training acc')
ax[0].plot(x, val_acc, 'r', label='Validation acc')
ax[0].set_title('Training and validation accuracy')
ax[0].legend()
ax[1].plot(x, loss, 'b', label='Training loss')
ax[1].plot(x, val_loss, 'r', label='Validation loss')
ax[1].set_title('Training and validation loss')
ax[1].legend()

plt.show()

#%%
pred = model3.predict(X_test)
pred = np.asarray([1 if x[0] >= 0.5 else 0 for x in pred])

conf_mat = confusion_matrix(Y_test, pred)
f1 = f1_score(Y_test, pred)