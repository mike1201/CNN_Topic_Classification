from data_helpers import *
from word2vec_CNN import *
import time
import sys

loaded_data, x_text_split = load_data_word2vec('data')
print("Loading data is done")
time.sleep(3)

vocabulary_size = 10000
x_text = loaded_data[0]
data, count, ordered_words = build_dataset(x_text, vocabulary_size)
print("building data set is done")
time.sleep(3)

# 2
np.random.seed(1)
tf.set_random_seed(1)
vocabulary_size = 10000
batch_size = 128
embedding_size = 128
skip_window = 2
num_skips = 4
num_sampled = 64
valid_size = 16
valid_window = 100

# 2-1
X_inputs = tf.placeholder(tf.int32, shape=[batch_size]) # X
Y_inputs = tf.placeholder(tf.int32, shape=[batch_size, 1]) # Y
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)


# 2-2
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
embed = tf.nn.embedding_lookup(embeddings, X_inputs)


# 2-3
nce_W = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                          stddev= 1.0 / math.sqrt(embedding_size))) # [3000, 128]
nce_b = tf.Variable(tf.zeros([vocabulary_size]))
nce_loss = tf.nn.nce_loss(weights=nce_W,
                          biases=nce_b,
                          labels=Y_inputs,
                          inputs=embed,
                          num_sampled=num_sampled,
                          num_classes=vocabulary_size)
loss = tf.reduce_mean(nce_loss)

# 2-4
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 2-5
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm # embedding vector normalize
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True) # cosine


# 3
num_steps = 500000

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    average_loss, data_index = 0, 0
    for step in range(num_steps):
        # make batch, labels, data_index
        batch_inputs, batch_labels, data_index = generate_batch(data, batch_size, num_skips, skip_window, data_index)
        _, loss_val = session.run([optimizer, loss],
                                  feed_dict={X_inputs: batch_inputs, Y_inputs: batch_labels})
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 5
            print('Average loss at step {} : {}'.format(step, average_loss))
            average_loss = 0

    final_embeddings = normalized_embeddings.eval()

print("embedding matrix is made")



embedding_zip = zip(final_embeddings, ordered_words)
main("data", embedding_zip)
print("embedded matrix and dictionary are  made")


f = open("data/word2vec_pickle", 'rb')
pickling = pickle.load(f)
embeds = pickling['embedding']
word_id_convert("data", x_text_split, loaded_data)
print("data_indexization is completed")



