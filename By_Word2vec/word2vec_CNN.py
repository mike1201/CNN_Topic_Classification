from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import collections
import math
import os
import zipfile
import sys
import pickle
import time


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


# 1-2
# 1-2-1
def build_dataset(words, n_words):
    unique = collections.Counter(words)
    orders = unique.most_common(n_words - 1)
    count = [['UNK', -1]]
    count.extend(orders)
    dictionary = {word: i for i, (word, _) in enumerate(count)}
    # dictionary = { (UNK, 0) (the, 1) (of, 2) (and, 3) (one, 4) (in, 5) (a, 6) (to, 7) }
    ordered_words = []
    for (word, _) in count:
        ordered_words.append(word)

    data = []
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            count[0][1] += 1
        data.append(index)              # data = [0, 1, 1640, 30, ...]

    return data, count, ordered_words


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
        data_index = (data_index + 1) % len(data)   # 다음 단어 인덱스로 이동. len(data) = 17005207

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


def main(data_dir, embedding_zip):
    f = open(data_dir+'/word2vec_pickle', 'wb')
    word_dict = {}
    wordvec =[]
    i = 0
    for embeds, word in embedding_zip: 
        word_dict[word] = i
        embeds = list(embeds)
        wordvec.append(embeds)
        i = i+1
    embedding = np.array(wordvec)
    
    pickling = {}
    pickling = {'embedding' : embedding, 'word_dict' : word_dict}
    pickle.dump(pickling,f)
    f.close()


# 1-(3)
def word_id_convert(data_dir, x_text_split, loaded_data):

    # load question data
    x_text = x_text_split
    y = loaded_data[1]
    max_document_length = max([len(x.split()) for x in x_text])


    # load embedded word
    h = open(data_dir + '/word2vec_pickle', 'rb')
    pickling = pickle.load(h)
    word_dict = pickling['word_dict']


    # [ques1, ques2, ...] --) [[word1, word2 ..] , [word1, word2, ...], [...] , ... ]
    splitter = [x.split() for x in x_text]
    word_indices = []


    for sentence in splitter: # sentence = [word1, word2, ... ]
        word_index = [word_dict[word] if word in word_dict else word_dict['the'] for word in sentence] # index
        padding = max_document_length -  len(word_index)
        padder = [2 for i in range(padding)]
        word_index = word_index + padder # [index1,index2, ... 2, 2, 2, ... ]
        word_indices.append(word_index)


    # Save index_question
    word_indices = np.array(word_indices)
    word_index_pickle = open(data_dir + '/word_index_pickle_word2vec', 'wb')
    pickling = {'word_indices': word_indices, 'y': y}
    pickle.dump(pickling, word_index_pickle)

'''

# 4
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'

    plt.figure(figsize=(18, 18))        # in inches

    for (x, y), label in zip(low_dim_embs, labels):
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)
    plt.show()

try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only])     # (500, 2)
    labels = ordered_words[:plot_only]

    plot_with_labels(low_dim_embs, labels)

except ImportError:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')

'''

