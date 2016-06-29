#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Clara Vania

from __future__ import unicode_literals, division
import numpy as np
import os
import re
import sys
import codecs
import random
import cPickle
import collections
import operator

class TextLoader:
    def __init__(self, args, train=True):
        self.save_dir = args.save_dir
        self.batch_size = args.batch_size
        self.num_steps = args.num_steps
        self.out_vocab_size = args.out_vocab_size
        self.words_vocab_file = os.path.join(self.save_dir, "words_vocab.pkl")

        if train:
            self.train_data = self.read_dataset(args.train_file)
            self.dev_data = self.read_dataset(args.dev_file)
            self.preprocess()
        else:
            self.load_preprocessed()

    def read_dataset(self, filename):
        """
        Read dataset from a file
        """
        data = list()
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.lower()
                for word in line.split():
                    if len(word) < 100 and 'www.' not in word and 'http' not in word:
                        data.append(word)
                data.append('<eos>')
        return data

    def preprocess(self):
        """
        Preprocess dataset and build vocabularies
        """
        self.word_to_id, self.unk_word_list = self.build_vocab(self.train_data)
        self.word_vocab_size = len(self.word_to_id)
        with codecs.open(self.words_vocab_file, 'w') as f:
            cPickle.dump((self.word_to_id, self.unk_word_list), f)

    def load_preprocessed(self):
        with codecs.open(self.words_vocab_file) as f:
            self.word_to_id, self.unk_word_list = cPickle.load(f)
            self.word_vocab_size = len(self.word_to_id)

    def build_vocab(self, train_data):
        """
        Build the vocabulary
        :param input_file: training data
        """
        unk_list = set()

        counter = collections.Counter(train_data)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[-1], x[0]))
        item_to_id = dict()

        if "<unk>" not in item_to_id:
            item_to_id['<unk>'] = len(item_to_id)

        for i, (x, y) in enumerate(count_pairs):
            if x not in item_to_id:
                item_to_id[x] = len(item_to_id)
            if y == 1:
                unk_list.add(x)
        return item_to_id, unk_list

    def data_to_word_ids(self, input_data, filter=False):
        """
        Given a list of words, convert each word into it's word id
        :param input_data: a list of words
        :return: a list of word ids
        """
        _buffer = list()
        for word in input_data:
            word = word.lower()
            # flag to randomize token with frequency one
            flag = 1
            if word in self.unk_word_list:
                flag = random.randint(0, 1)
            if word in self.word_to_id and flag == 1:
                # if filter is True, reduce output vocabulary for softmax
                # (map words not in top self.out_vocab_size to UNK)
                if filter:
                    if self.word_to_id[word] <= self.out_vocab_size:
                        _buffer.append(self.word_to_id[word])
                    else:
                        _buffer.append(self.word_to_id['<unk>'])
                else:
                    _buffer.append(self.word_to_id[word])
            else:
                _buffer.append(self.word_to_id['<unk>'])
        return _buffer

    def encode_data(self, data):
        """
        Encode data according to the specified encoding
        :param data: input data
        :return: encoded input data
        """
        data = self.data_to_word_ids(data)
        return data

    def data_iterator(self, raw_data, batch_size, num_steps):
        data_len = len(raw_data)
        batch_len = data_len // batch_size
        data = []
        for i in range(batch_size):
            x = raw_data[batch_len * i:batch_len * (i + 1)]
            data.append(x)

        epoch_size = (batch_len - 1) // num_steps

        if epoch_size == 0:
            raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

        for i in range(epoch_size):
            xs = list()
            ys = list()
            for j in range(batch_size):
                x = data[j][i*num_steps:(i+1)*num_steps]
                y = data[j][i*num_steps+1:(i+1)*num_steps+1]
                xs.append(self.encode_data(x))
                ys.append(self.data_to_word_ids(y, True))
            yield (xs, ys)