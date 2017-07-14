import tensorflow as tf
import collections
import numpy as np
import random
import re
import math

'''
Get word embedding vector
'''
f = open("data/segTrain.txt", "r")
vocalbuary = f.read().split()
f.seek(0)
questions = []
answers = []
while True:
  line = f.readline()
  if not line:
    break
  line = line.split('\t')
  questions.append(["<GO>"] + line[0].split() + ["<EOS>"])
  answers.append(["<GO>"] + line[1].split() + ["<EOS>"])
f.close()

vocabulary_size = 5000
vocabulary_size += 2 #for <GO> and <EOS> 

def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(vocalbuary,
                                                            vocabulary_size)
#reversed_dictionary: int -> word
#dictionary: word -> int
'''
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
'''
del vocalbuary, count

data_index = 0

# Step 3: Function to generate a training batch for the skip-gram model.

'''
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels
'''

'''
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i,batch_size = 128]])
        '''

batch_size = 128
embedding_size = 512  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label. 0]])
num_sampled = 64    # Number of negative examples to sample.

train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

# Ops and variables pinned to the CPU because of missing GPU implementation
with tf.device('/cpu:0'):
  # Look up embeddings for inputs.
  embeddings = tf.Variable(
      tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
  '''
  embed = tf.nn.embedding_lookup(embeddings, train_inputs)

  # Construct the variables for the NCE loss
  nce_weights = tf.Variable(
      tf.truncated_normal([vocabulary_size, embedding_size],
                          stddev=1.0 / math.sqrt(embedding_size)))
  nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
'''

#Now we get the word embedding vector, start building biRNN
#First translate questions and answers to predefined integer
questions_data = []
answers_data = []
questions_data_length = []
answers_data_length = []
for question in questions:
  words = []
  for d in question:
    try:
      words.append(dictionary[d])
    except:
      words.append(dictionary['UNK'])
  questions_data.append(words)
  questions_data_length.append([len(words)])
for answer in answers:
  words = []
  for d in answer:
    try:
      words.append(dictionary[d])
    except:
      words.append(dictionary['UNK'])
  answers_data.append(words)
  answers_data_length.append([len(words)])

#Now define a batch generation function
#it will return batch_size of q&a batch, sequence_length
#!!!!!!!Not throughtly tested yet, may produce bug
current_qa_index = 0
qa_pairs_count = len(questions)

def generate_qa_batch(qa_batch_size):
  global current_qa_index, questions_data, answers_data, questions_data_length, answers_data_length
  if current_qa_index > qa_pairs_count - 1:
    current_qa_index = 0
  target_qa_index = current_qa_index + qa_batch_size
  if target_qa_index > qa_pairs_count:
    qa_batch_size = qa_pairs_count - current_qa_index
    target_qa_index = qa_pairs_count
  batch_q = questions_data[current_qa_index:target_qa_index]
  batch_a = answers_data[current_qa_index:target_qa_index]
  batch_q_len = questions_data_length[current_qa_index:target_qa_index]
  batch_a_len = answers_data_length[current_qa_index:target_qa_index]
  #pad short sequences
  max_q_len = max(batch_q_len, key=lambda x:x[0])[0]
  max_a_len = max(batch_a_len, key=lambda x:x[0])[0]
  for i in range(len(batch_q)):
    currentLen = len(batch_q[i])
    if currentLen < max_q_len:
      for _ in range(max_q_len - currentLen):
        batch_q[i].append(0) 
  for i in range(len(batch_a)):
    currentLen = len(batch_a[i])
    if currentLen < max_a_len:
      for _ in range(max_a_len - currentLen):
        batch_a[i].append(0) 
  current_qa_index = target_qa_index
  return batch_q, batch_a, batch_q_len, batch_a_len, qa_batch_size
'''
#for test only
a, b, c, d, e = generate_qa_batch(2)
print(a, b, c, d, e)
exit()
'''

qa_batch_size = 128
train_times = 10000
q_placeholder = tf.placeholder(tf.int32, shape=(None, None), name="q_placeholder")
q_sequence_length = tf.placeholder(tf.int32, shape=(None, 1), name="q_sequence_length")
a_placeholder = tf.placeholder(tf.int32, shape=(None, None), name="a_placeholder")
a_sequence_length = tf.placeholder(tf.int32, shape=(None, 1), name="a_sequence_length")
real_batch_size_placeholder = tf.placeholder(tf.int32, name="real_batch_size_placeholder")
with tf.variable_scope("qRNNfw"):
  q_fw_cell = tf.nn.rnn_cell.GRUCell(embedding_size)
with tf.variable_scope("qRNNbw"):
  q_bw_cell = tf.nn.rnn_cell.GRUCell(embedding_size)
with tf.variable_scope("aRNNfw"):
  a_fw_cell = tf.nn.rnn_cell.GRUCell(embedding_size)
with tf.variable_scope("aRNNbw"):
  a_bw_cell = tf.nn.rnn_cell.GRUCell(embedding_size)
q_embedded = tf.nn.embedding_lookup(embeddings, q_placeholder)
a_embedded = tf.nn.embedding_lookup(embeddings, a_placeholder)
with tf.variable_scope("qRNN"):
  hx, output_states = tf.nn.bidirectional_dynamic_rnn(
      q_fw_cell, q_bw_cell, q_embedded, sequence_length=tf.squeeze(q_sequence_length, 1), dtype=tf.float32)
with tf.variable_scope("aRNN"):
  hy, _ = tf.nn.bidirectional_dynamic_rnn(
      a_fw_cell, a_bw_cell, a_embedded, sequence_length=tf.squeeze(a_sequence_length, 1), dtype=tf.float32)
hx = (hx[0] + hx[1]) / 2
hy = (hy[0] + hy[1]) / 2
hx = tf.reduce_sum(hx, reduction_indices=1)
hx = tf.div(hx, tf.cast(q_sequence_length, tf.float32))
hy = tf.reduce_sum(hy, reduction_indices=1)
hy = tf.div(hy, tf.cast(a_sequence_length, tf.float32))
hxhy = tf.concat([hx, hy], 1)

#Get mean and variance of latent variable z
z_hidden_size = 64
mean = tf.contrib.layers.fully_connected(hxhy, z_hidden_size);
sigma = tf.exp(tf.contrib.layers.fully_connected(hxhy, z_hidden_size));
distribution = tf.contrib.distributions.MultivariateNormalDiag(loc=mean, scale_diag=sigma)
z = distribution.sample()

#provide esstential information for training decoder
hxz = tf.concat([hx, z], 1)
with tf.variable_scope("decoderRNNCell"):
  decoder_cell = tf.nn.rnn_cell.GRUCell(embedding_size + z_hidden_size)
with tf.variable_scope("decoderRNN"):
  result, _ = tf.nn.dynamic_rnn(
      decoder_cell, a_embedded, sequence_length=tf.squeeze(a_sequence_length, 1), initial_state=hxz)
result = tf.contrib.layers.fully_connected(result, vocabulary_size)
square_sigma = tf.square(sigma)
square_mean = tf.square(mean)
a_target = tf.pad(a_placeholder[:][1:], [[0, 0], [0, 1]])
loss = 0.5 * tf.reduce_sum(1 + tf.log(square_sigma) - square_mean - square_sigma) / tf.cast(real_batch_size_placeholder, tf.float32)
loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=result, labels=tf.one_hot(a_target, vocabulary_size, dtype=tf.float32)))
train = tf.train.AdamOptimizer().minimize(loss)


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  #Get Word Embedding Vector
  '''
  average_loss = 0
  for step in xrange(10001):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0
  '''
  
  #get bRNN
  for _ in range(1):
    batch_q, batch_a, batch_q_len, batch_a_len, real_qa_batch_size = generate_qa_batch(
        qa_batch_size)
    loss = sess.run([loss], feed_dict={q_placeholder: batch_q, a_placeholder: batch_a,
                                  q_sequence_length: batch_q_len, a_sequence_length: batch_a_len, real_batch_size_placeholder:real_qa_batch_size})
    print(loss)
