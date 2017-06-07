import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU
from keras.layers import Concatenate, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau

import sys

from time import time
import gc
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from itertools import chain
from collections import Counter
from gensim.models import word2vec
import pickle


### useful functions
def add_Dense(merged, num_dense, activation='sigmoid'):
    merged = Dense(num_dense, activation=activation)(merged)
    merged = BatchNormalization()(merged)
    return merged

def add_Dropout(merged, rate_drop):
    merged = Dropout(rate_drop)(merged)
    return merged

def add_lstm(merged, num_lstm, dropout, recurrent_dropout):
    merged = LSTM(num_lstm, dropout=dropout, recurrent_dropout=recurrent_dropout)(merged)
    return merged

def add_lstm1(merged, num_lstm, dropout, recurrent_dropout):
    merged = LSTM(num_lstm, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True)(merged)
    return merged

def add_gru(merged, num_lstm, dropout, recurrent_dropout):
    merged = GRU(num_lstm, dropout=dropout, recurrent_dropout=recurrent_dropout)(merged)
    return merged


myaddfunc = {'dropout':add_Dropout, 'dense':add_Dense, 'lstm':add_lstm, 'gru':add_gru, 'lstm1':add_lstm1}

def create_NN(dict_struct, optimizer='nadam', epochs=1, batch_size=2048, drop_rate=0.2, num_dense=100, activation='tanh',
              firts_lstm={'num_lstm':190, 'rate_drop_lstm':0.4} ):

    print('\n', dict_struct, '\n')
    is_lstm = 'lstm' in dict_struct.keys()
    is_recurrent = ('lstm' in dict_struct.keys()) or ('gru' in dict_struct.keys()) ## qui salva i nomi per tutti non va bene
    other_params = '_'.join([activation, str(batch_size), str(drop_rate)]) + '_' + str(firts_lstm['num_lstm']) + '_' + str(firts_lstm['rate_drop_lstm'])
    STAMP = '_'.join( list( dict_struct.keys() ) ) + '_' + optimizer + '_' + str(is_lstm) +'_' + other_params
    print('\n', STAMP, '\n')
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    rate_drop_lstm = firts_lstm['rate_drop_lstm']
    lstm_layer = LSTM(firts_lstm['num_lstm'], dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm, 
                      return_sequences=is_recurrent)
#    gru_layer = GRU(firts_lstm['num_lstm'], dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm,
#                   return_sequences=is_recurrent)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, 
                                  patience=5, min_lr=0.00005, verbose=1)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    
    if is_lstm:
        x = lstm_layer(embedded_sequences_1)
        y = lstm_layer(embedded_sequences_2)
    else:
        pass
#        x = gru_layer(embedded_sequences_1)
#        y = gru_layer(embedded_sequences_2)
    
    # my structure
    only_once = True
    if len(dict_struct) == 0:
        merged = concatenate([x, y])
    for k in dict_struct:
        if k in ('lstm', 'gru', 'lstm1'):
            x = myaddfunc[k](x, **dict_struct[k])
            y = myaddfunc[k](y, **dict_struct[k])
            x = add_Dropout(x, drop_rate)
            y = add_Dropout(y, drop_rate)
            if (len(dict_struct) == 1):
                merged = concatenate([x, y])
            continue
        if k not in ('lstm', 'gru', 'lstm1') and only_once:
            merged = concatenate([x, y])
            add_Dropout(merged, drop_rate)
            merged = BatchNormalization()(merged)
            only_once = False
        merged = myaddfunc[k](merged, **dict_struct[k])
        
        
    merged = Dense(num_dense, activation=activation)(merged)
    merged = Dropout(drop_rate)(merged)
    merged = BatchNormalization()(merged)    
    preds = Dense(1, activation='sigmoid')(merged)
       
    
    model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
    print(model.summary())
    print(STAMP)

    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    bst_model_path = STAMP + '.h5'
    model_checkpoint = ModelCheckpoint(
        bst_model_path, save_best_only=True, save_weights_only=True, period=1, monitor='val_loss')
                
    csv_logger = CSVLogger('{}.log'.format(STAMP))
    hist = model.fit(
        [data_1_train, data_2_train],
        labels_train,
        validation_data=([data_1_val, data_2_val], labels_val, weight_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        class_weight=class_weight,
        callbacks=[early_stopping, model_checkpoint, csv_logger, reduce_lr])
    
    # model.load_weights(bst_model_path)
    model.save(bst_model_path)
    return (STAMP, min(hist.history['val_loss']))



### set global parameters
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 135
VALIDATION_SPLIT = 0.13
BASE_DIR = ''
EMBEDDING_FILE = BASE_DIR + 'model_best.bin'


labels = np.loadtxt('labels.gz')
data_1 = np.loadtxt('data_1.gz')
data_2 = np.loadtxt('data_2.gz')
embedding_matrix = np.loadtxt('embedding_matrix.gz')        
        
np.random.seed(1234)
perm = np.random.permutation(len(data_1))
idx_train = perm[:int(len(data_1)*(1-VALIDATION_SPLIT))]
idx_val = perm[int(len(data_1)*(1-VALIDATION_SPLIT)):]

data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
labels_train = np.concatenate((labels[idx_train], labels[idx_train]))

data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))
labels_val = np.concatenate((labels[idx_val], labels[idx_val]))

weight_val = np.ones(len(labels_val))
r = 1./0.165
rtrain = (len(labels_val) - sum(labels_val))/sum(labels_val)
weight_val *= 0.282
# 0.472001959
weight_val[labels_val==0] = 1.
#r*rtrain
# 1.309028344
re_weight = True



### create some NN, USE SYS HERE TO PASS THE MYNET YOU WANT TO TRAIN

import time
np.random.seed( int(time.time()) )
from collections import OrderedDict
num_lstm = 210 # int(np.random.randint(175, 230)) # 175-275
rate_drop_lstm = 0.3 # 0.15 + np.random.rand() * 0.25
num_dense = 135 # np.random.randint(100, 150)
class_weight = {0:1, 1:0.282} #{0: r*rtrain, 1: 1}# {0: 1.309028344, 1: 0.472001959}
firts_lstm={'num_lstm':num_lstm, 'rate_drop_lstm':rate_drop_lstm}

mynet0 = OrderedDict()

mynet1 = OrderedDict()
mynet1['lstm'] = {'num_lstm':num_lstm//2, 'dropout':rate_drop_lstm, 'recurrent_dropout':rate_drop_lstm}

mynet2 = OrderedDict()
mynet2['dense'] = {'num_dense':num_dense, 'activation':'sigmoid'}
mynet2['dropout'] = {'rate_drop':0.15 + np.random.rand() * 0.25}

mynet3 = OrderedDict()
mynet3['lstm'] = {'num_lstm':num_lstm//2, 'dropout':rate_drop_lstm, 'recurrent_dropout':rate_drop_lstm}
mynet3['dense'] = {'num_dense':num_dense, 'activation':'sigmoid'}
mynet3['dropout'] = {'rate_drop':0.15 + np.random.rand() * 0.25}

mynet4 = OrderedDict()
mynet4['gru'] = {'num_lstm':num_lstm//2, 'dropout':rate_drop_lstm, 'recurrent_dropout':rate_drop_lstm}

mynet5 = OrderedDict()
mynet5['gru'] = {'num_lstm':num_lstm//2, 'dropout':rate_drop_lstm, 'recurrent_dropout':rate_drop_lstm}
mynet5['dense'] = {'num_dense':num_dense, 'activation':'sigmoid'}
mynet5['dropout'] = {'rate_drop':0.15 + np.random.rand() * 0.25}


mynet3l = OrderedDict()
mynet3l['lstm1'] = {'num_lstm':num_lstm//2, 'dropout':rate_drop_lstm, 'recurrent_dropout':rate_drop_lstm}
mynet3l['lstm'] = {'num_lstm':num_lstm//4, 'dropout':rate_drop_lstm, 'recurrent_dropout':rate_drop_lstm}
mynet3l['dense'] = {'num_dense':num_dense, 'activation':'sigmoid'}
mynet3l['dropout'] = {'rate_drop':0.15 + np.random.rand() * 0.25}


activation=sys.argv[2]
drop_rate=float(sys.argv[4])

mynet_final = OrderedDict()
mynet_final['lstm'] = {'num_lstm':num_lstm//2, 'dropout':rate_drop_lstm, 'recurrent_dropout':rate_drop_lstm}
mynet_final['dense'] = {'num_dense':num_dense, 'activation':activation}
mynet_final['dropout'] = {'rate_drop':drop_rate}

nns = {'mynet0':mynet0, 'mynet1':mynet1, 'mynet2':mynet2, 'mynet3':mynet3,
       'mynet4':mynet4, 'mynet5':mynet5, 'mynet3l':mynet3l, 'mynet_final':mynet_final}
nb_words = min(MAX_NB_WORDS, 111800) + 1


create_NN(nns[sys.argv[1]], optimizer='rmsprop', epochs=100, batch_size=int(sys.argv[3]), activation=activation,
          drop_rate=drop_rate, num_dense=num_dense, firts_lstm=firts_lstm)

                            




