import os
import time
import pickle
import sys
import numpy as np
import argparse


# 1-2
def main(data_dir):
    f = open(data_dir + '/glove.6B.50d.txt', 'r', encoding='EUC-KR') # Be careful of Encoding
    g = open(data_dir + '/glove.6B.50d_pickle', 'wb')

    
    word_dict = {}
    wordvec = []

    
    for idx, line in enumerate(f.readlines()):
        word_split = line.split()  # line = [the, 0.354, 0.213 .... ]         
        
        d = word_split[1:] 
        d[-1] = d[-1][:-1] # d = embedding numbers

        try:
            d = [float(e) for e in d]
        except ValueError:
            print(word, 'is corrupt')
            continue
        
        wordvec.append(d) #  [ word1_nums, word2_nums, ... ]
                          #  word_nums = [embed1, embed2, ... ]
        
        word = word_split[0]
        word = str(word)
        word_dict[word] = idx         

    embedding = np.array(wordvec)

    pickling = {}
    pickling = {'embedding' : embedding, 'word_dict': word_dict} 
    pickle.dump(pickling, g)
    f.close()
    g.close()


# 1-3
def word_id_convert(data_dir):

    # load question data
    g = open(data_dir + '/data_pickling', 'rb')
    pickling = pickle.load(g)
    x_text = pickling['x']
    y = pickling['y']
    max_document_length = max([len(x.split()) for x in x_text])


    # load embedded word
    h = open(data_dir + '/glove.6B.50d_pickle', 'rb')
    pickling = pickle.load(h)
    word_dict = pickling['word_dict']


    splitter = [x.split() for x in x_text]
    # x_text = [ques1, ques2, ...]
    # splitter = [ [word1, word2 ..], [word1, word2, ...], [...] , ... ]
    word_indices = []
    

    for sentence in splitter: # sentence = [word1, word2, ... ]
        word_index = [word_dict[word] if word in word_dict else word_dict['the'] for word in sentence] # index
        padding = max_document_length -  len(word_index)
        padder = [2 for i in range(padding)]
        word_index = word_index + padder # [index1,index2, ... 2, 2, 2, ... ]
        word_indices.append(word_index)

    # Save index_question
    word_indices = np.array(word_indices)
    word_index_pickle = open(data_dir + '/word_index_pickle', 'wb')
    pickling = {'word_indices': word_indices, 'y': y}
    pickle.dump(pickling, word_index_pickle)


def write_concat_vec(data_dir):
    word_index_pickle = open(data_dir + '/word_index_pickle', 'rb')
    pickling = pickle.load(word_index_pickle)
    x = pickling['word_indices']
    y = pickling['y']
    g = open(data_dir + '/glove.6B.50d_pickle', 'rb')
    pickling = pickle.load(g)
    embedding = pickling['embedding']
    
    l = []
    m = []
    arr = np.array([0,1,2,3,4])
    for idx, sentence in enumerate(x):
        concater = np.array([])
    
        # [word1, word2, ... ] --) [embed11, embed12, ... embed1n, embed21, ... ... embed ]
        sentence_vec = embedding[sentence].flatten()
        sentence_vec = np.reshape(sentence_vec, (1, sentence_vec.shape[0])) # 1 x 2050

        # print sentence_vec.shape
        l.append(sentence_vec)
        label = np.sum(y[idx] * arr)
        m.append(label)

    concater = np.squeeze(np.array(l)) # remove single dimension
    m = np.array(m)

    h = open(data_dir + '/concat_vec_labels', 'wb')
    pickling = {'x': concater, 'y': m}
    pickle.dump(pickling, h, protocol = 2)
    h.close()
    g.close()
    word_index_pickle.close()
    # print concater.shape

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='/home/hw/data',
		           help='data directory containing glove vectors')
	args = parser.parse_args()
	data_dir = args.data_dir
	
	main(data_dir)
	#oad_pickle()
	word_id_convert(data_dir)
	#load_data()
	write_concat_vec(data_dir)
