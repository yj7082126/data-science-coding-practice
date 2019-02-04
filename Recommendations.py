# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 12:47:05 2019

@author: user
"""

import numpy as np
import pandas as pd
import gzip
import spacy
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, Activation, Embedding, Flatten, merge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
df = df[['user_id', 'item_id', 'score']]

#%%
X = np.asarray(df[['user_id', 'item_id']])
Y = np.asarray(df['score'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train, test_size=(1/9), random_state=42)

#%%
train, test = train_test_split(df, test_size=0.1, random_state=42)
train, dev = train_test_split(df, test_size=(1/9), random_state=42)

X_train = [np.array([[element] for element in train.user_id]), 
           np.array([[element] for element in train.item_id])]
Y_train = np.array([[element] for element in train.score])

X_dev = [np.array([[element] for element in dev.user_id]), 
         np.array([[element] for element in dev.item_id])]
Y_dev = np.array([[element] for element in dev.score])

X_test = [np.array([[element] for element in test.user_id]), 
         np.array([[element] for element in test.item_id])]
Y_test = np.array([[element] for element in test.score])
#%%
embed = 50
n_user=len(df['user_id'].unique())
n_item=len(df['item_id'].unique())

#%%
user_input = Input(shape=[1], name='user_input')
user_embed = Embedding(n_user, embed, name='user_embed')(user_input)
user_flat = Flatten(name='user_flat')(user_embed)
#user_drop = Dropout(0.2, name='user_drop')(user_flat)

item_input = Input(shape=[1], name='item_input')
item_embed = Embedding(n_item, embed, name='item_embed')(item_input)
item_flat = Flatten(name='item_flat')(item_embed)
#item_drop = Dropout(0.2, name='item_drop')(item_flat)

prod = merge.concatenate([user_flat, item_flat], name='DotProduct')
prod = Dropout(0.5)(prod)
x = Dense(64, activation='relu')(prod)
y = Dense(1)(x)

model = Model([user_input, item_input], y)

model.compile(loss='mae', optimizer='adam', metrics=['acc'])

#%%
model.fit(X_train, Y_train, batch_size=40, epochs=10, 
           validation_data = (X_dev, Y_dev), shuffle=True)

pred = model.predict(X_test)
pred = np.round(pred)

accuracy_score(Y_test, pred)
