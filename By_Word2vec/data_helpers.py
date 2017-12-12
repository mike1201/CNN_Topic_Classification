import numpy as np
import re
import os
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


# 1-1
def load_data_word2vec(data_dir):
    one_hot_vector = [0,0,0,0,0]
    topics = ['Technology', 'Business', 'Food', 'Design', 'Books']
    str = "the"
    x_text_all = []
    x_text_split = []
    y = []

    for idx, topic in enumerate(topics):
        # made x
        clean_questions = list(open(data_dir + '/' + topic + 'clean_question.txt', mode='r').readlines())
        clean_questions = [s.strip() for s in clean_questions]
        for line in clean_questions:
            words = line.split()
            x_text_all = x_text_all + words

        
        x_text_split = x_text_split + clean_questions
        
        
        # made y
        if topic == 'Technology':
            y = y + [[1,0,0,0,0] for _ in clean_questions]
        elif topic == 'Business':
            y = y + [[0,1,0,0,0] for _ in clean_questions]
        elif topic == 'Food':
            y = y + [[0,0,1,0,0] for _ in clean_questions]
        elif topic == 'Design':
            y = y + [[0,0,0,1,0] for _ in clean_questions]
        elif topic == 'Books':
            y = y + [[0,0,0,0,1] for _ in clean_questions]        # print labels
        one_hot_vector[idx] = 0
    y = np.array(y)
    return [x_text_all, y], x_text_split


# 1-(4)
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1     
    for epoch in range(num_epochs):

        if shuffle: # Shuffle the data at each epoch
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
            # np.arange(3) = array([0,1,2]) , np.random.permutation() : randomly shuffle

        else: 
            shuffled_data = data

        # make batches at each epoch
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]





# 1-3
# 1-3-1
def generate_batch(data, batch_size, num_skips, skip_window, data_index):
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    # 1-3-1-1
    span = 2 * skip_window + 1
    assert span > num_skips

    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)   # len(data) = 17005207

    # 1-3-1-2
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    for i in range(batch_size // num_skips):

        # make batch(=X)
        start = i * num_skips  # 0*2=0
        batch[start:start + num_skips] = buffer[skip_window]

        # make labels(=Y)
        targets = list(range(span))
        targets.pop(skip_window)
        np.random.shuffle(targets)
        for j in range(num_skips):
            labels[start + j, 0] = buffer[targets[j]]


        buffer.append(data[data_index])  # [data[0], data[1], data[2]] --) [ data[1], data[2], data[3] ]
        data_index = (data_index + 1) % len(data)  # data_index = 3+1 = 4


    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels, data_index




