#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import os
import pickle
from six.moves import urllib

import tflearn
from tflearn.data_utils import *

path = "data/dnd-monsters.txt"
char_idx_file = 'data/monster_idx.pickle'

if not os.path.isfile(path):
    urllib.request.urlretrieve("https://s3-ap-southeast-2.amazonaws.com/tf-training-data/dnd-monsters.txt", path)

maxlen = 25

char_idx = None
if os.path.isfile(char_idx_file):
  print('Loading previous monster_idx')
  char_idx = pickle.load(open(char_idx_file, 'rb'))

X, Y, char_idx = \
    textfile_to_semi_redundant_sequences(path, seq_maxlen=maxlen, redun_step=3,
                                         pre_defined_char_idx=char_idx)

pickle.dump(char_idx, open(char_idx_file,'wb'))

g = tflearn.input_data([None, maxlen, len(char_idx)])
g = tflearn.lstm(g, 512, return_seq=True)
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, 512, return_seq=True)
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, 512)
g = tflearn.dropout(g, 0.5)
g = tflearn.fully_connected(g, len(char_idx), activation='softmax')
g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy',
                       learning_rate=0.001)

m = tflearn.SequenceGenerator(g, dictionary=char_idx,
                              seq_maxlen=maxlen,
                              clip_gradients=5.0,
                              checkpoint_path='data/model_monsters')

for i in range(50):
    num_test_chars = 100
    seed = random_sequence_from_textfile(path, maxlen)
    m.fit(X, Y, validation_set=0.1, batch_size=128,
          n_epoch=1, run_id='monsters')
    print("-- TESTING...")
    print("-- Test with temperature of 3.0 --")
    print(m.generate(num_test_chars, temperature=3.0, seq_seed=seed))
    print("-- test with temperature of 2.0 --")
    print(m.generate(num_test_chars, temperature=2.0, seq_seed=seed))
    print("-- test with temperature of 1.0 --")
    print(m.generate(num_test_chars, temperature=1.0, seq_seed=seed))
    print("-- test with temperature of 0.5 --")
    print(m.generate(num_test_chars, temperature=0.5, seq_seed=seed))
