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
'''
vocabulary = f.read().split()
f.seek(0)
'''
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
vocabulary = []
for q in questions+answers:
  vocabulary += q

vocabulary_size = 5000
vocabulary_size += 2  # for <GO> and <EOS>


def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
    '''
  dictionary["<GO>"] = len(dictionary)
  dictionary["<EOS>"] = len(dictionary)
  '''
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


data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)
#reversed_dictionary: int -> word
#dictionary: word -> int
'''
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
'''
del vocabulary, count

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

embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

# Ops and variables pinned to the CPU because of missing GPU implementation
'''
with tf.device('/cpu:0'):
  # Look up embeddings for inputs.
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
  max_q_len = max(batch_q_len, key=lambda x: x[0])[0]
  max_a_len = max(batch_a_len, key=lambda x: x[0])[0]
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
#for testing only
a, b, c, d, e = generate_qa_batch(2)
print(a, b, c, d, e)
exit()
'''

qa_batch_size = 128
train_times = 2001
current_step = tf.placeholder(tf.float32)

q_placeholder = tf.placeholder(
    tf.int32, shape=(None, None), name="q_placeholder")
q_sequence_length = tf.placeholder(
    tf.int32, shape=(None, 1), name="q_sequence_length")
a_placeholder = tf.placeholder(
    tf.int32, shape=(None, None), name="a_placeholder")
a_sequence_length = tf.placeholder(
    tf.int32, shape=(None, 1), name="a_sequence_length")
real_batch_size_placeholder = tf.placeholder(
    tf.int32, name="real_batch_size_placeholder")
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
  _, hx = tf.nn.bidirectional_dynamic_rnn(
      q_fw_cell, q_bw_cell, q_embedded, sequence_length=tf.squeeze(q_sequence_length, 1), dtype=tf.float32)
with tf.variable_scope("aRNN"):
  _, hy = tf.nn.bidirectional_dynamic_rnn(
      a_fw_cell, a_bw_cell, a_embedded, sequence_length=tf.squeeze(a_sequence_length, 1), dtype=tf.float32)
hx = tf.concat([hx[0], hx[1]], 1)
hx = tf.contrib.layers.fully_connected(hx, embedding_size, activation_fn=tf.tanh)
hy = tf.concat([hy[0], hy[1]], 1)
hy = tf.contrib.layers.fully_connected(hy, embedding_size, activation_fn=tf.tanh)
# mean = tf.contrib.layers.fully_connected(tf.concat(hx, z_hidden_size, activation_fn=None)
'''
hx = tf.reduce_sum(hx, reduction_indices=1)
hx = tf.div(hx, tf.cast(q_sequence_length, tf.float32))
hy = tf.reduce_sum(hy, reduction_indices=1)
hy = tf.div(hy, tf.cast(a_sequence_length, tf.float32))
'''
hxhy = tf.concat([hx, hy], 1)
#Get mean and variance of latent variable z
z_hidden_size = 64
mean = tf.contrib.layers.fully_connected(hxhy, z_hidden_size, activation_fn=None)
sigma = tf.exp(tf.contrib.layers.fully_connected(hxhy, z_hidden_size, activation_fn=None) / 2.0) #div 2 -> sqrt
'''
distribution = tf.contrib.distributions.MultivariateNormalDiag(
    loc=tf.zeros(shape=(z_hidden_size)), scale_diag=tf.ones(shape=(z_hidden_size)))
'''
distribution = tf.random_normal([real_batch_size_placeholder, z_hidden_size])
z = tf.multiply(distribution, sigma) + mean  # dot multiply

#provide esstential information for training decoder
hxz = tf.concat([hx, z], 1)
hxz = tf.nn.dropout(hxz, 0.5)
with tf.variable_scope("decoderRNNCell"):
  decoder_cell = tf.nn.rnn_cell.GRUCell(embedding_size + z_hidden_size)
with tf.variable_scope("decoderRNN"):
  result, _ = tf.nn.dynamic_rnn(
      decoder_cell, a_embedded, sequence_length=tf.squeeze(a_sequence_length, 1), initial_state=hxz)
result = tf.contrib.layers.fully_connected(result, vocabulary_size, scope="rnn/resEmbedding", activation_fn=None)

def weight_cal_false():
  return tf.cond(tf.less(current_step, train_times * 0.9), lambda: (-0.5 / (0.8 * train_times) * (current_step - 0.1 * train_times)), lambda: tf.cast(0, tf.float32))

#calculate loss function
square_sigma = tf.square(sigma)
square_mean = tf.square(mean)
a_target = tf.pad(a_placeholder[:, 1:], [[0, 0], [0, 1]])
loss = 0.5 * tf.reduce_mean(1 + tf.log(square_sigma) - square_mean -
                           square_sigma)
kl_annealing_weight = tf.cond(tf.less(
    current_step, train_times / 10), lambda: tf.cast(0.5, tf.float32), weight_cal_false);
# kl_annealing_weight = 1
loss *= kl_annealing_weight
kl_lower_bound = 0.0002 #Level of KL loss at which to stop optimizing for KL.
loss = tf.cond(tf.less(loss, kl_lower_bound), lambda: kl_lower_bound, lambda: loss)
loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=result,
                                                               labels=tf.one_hot(a_target, vocabulary_size, dtype=tf.float32)))
train = tf.train.AdamOptimizer().minimize(loss)

saver = tf.train.Saver()

#now let's predict
'''
def predict_loop_body(last_output, last_state, output, i):
  global decoder_cell, embedding_size, z_hidden_size, vocabulary_size, embeddings
  last_output = tf.nn.embedding_lookup(embeddings, last_output)
  last_output = tf.convert_to_tensor(last_output)
  last_state = tf.convert_to_tensor(last_state)
  res = decoder_cell.call(last_output, last_state)
  ret = tf.contrib.layers.fully_connected(res[0], vocabulary_size, scope="resEmbedding", reuse=True, activation_fn=None)
  ret = tf.convert_to_tensor([tf.argmax(ret, axis=1)[0]])
  output = tf.concat([output, [tf.cast(ret, tf.int32)]], 1)
  i += 1
  return ret, res[1], output, i

def predict_loop_cond(last_output, last_state, output, i):
  if last_output[0] == dictionary["<EOS>"]:
    return tf.False
  return tf.less(i, 20)
'''

max_response_length = 20 #define the maximum response length
predict_batch_size = 1
go_embedded = tf.nn.embedding_lookup(embeddings, tf.tile([dictionary["<GO>"]], [predict_batch_size]))
eos_embedded = tf.nn.embedding_lookup(embeddings, tf.tile([dictionary["<EOS>"]], [predict_batch_size]))

def loop_fn(time, previous_output, previous_state, previous_loop_state):
  if previous_state is None:
    assert previous_output is None and previous_state is None
    return loop_fn_initial()
  else:
    return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

def loop_fn_initial():
  global embeddings, dictionary, max_response_length, predict_hxz, go_embedded
  element_finished = (0 >= max_response_length)
  return (element_finished, go_embedded, predict_hxz, None, None)

def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
  global max_response_length, embeddings, eos_embedded, embedding_size
  def get_next_input():
    '''
    predict_W = tf.get_variable("resEmbedding/weights")
    predict_b = tf.get_variable("resEmbedding/biases")
    output_logits = tf.matmul(predict_W, previous_output) + predict_b
    '''
    output_logits = tf.contrib.layers.fully_connected(previous_output, vocabulary_size, activation_fn=None, scope="resEmbedding", reuse=True)
    predict = tf.argmax(output_logits, axis=1)
    ret = tf.nn.embedding_lookup(embeddings, predict)
    ret = tf.reshape(ret, [predict_batch_size, embedding_size]) #just for skipping while_loop's shape checking
    return ret
  element_finished = (time >= max_response_length)
  finished = tf.reduce_all(element_finished)
  next_input = tf.cond(finished, lambda: eos_embedded, get_next_input)
  return (element_finished, next_input, previous_state, previous_output, None)

predict_q = q_placeholder[0:1, :]
predict_q_embedded = tf.nn.embedding_lookup(embeddings, predict_q)
predict_q_sequence_length = q_sequence_length[0:1, :]
'''
with tf.variable_scope("qRNN"):
  predict_hx, output_states = tf.nn.bidirectional_dynamic_rnn(
      q_fw_cell, q_bw_cell, predict_q_embedded, sequence_length=tf.squeeze(predict_q_sequence_length, 1), dtype=tf.float32, reuse=True)
predict_hx = (predict_hx[0] + predict_hx[1]) / 2
predict_hx = tf.reduce_sum(predict_hx, reduction_indices=1)
predict_hx = tf.div(predict_hx, tf.cast(predict_q_sequence_length, tf.float32))
'''
predict_hx = hx[0:1, :]
predict_z = tf.random_normal([1, z_hidden_size])
predict_hxz = tf.concat([predict_hx, predict_z], 1)
(predict_output, _, _) = tf.nn.raw_rnn(decoder_cell, loop_fn)
predict_output = predict_output.stack()
#now predict_output has shape [max_response_length, predict_batch_size, embedding_size+z_hidden_size]
# predict_output_shape = tf.shape(predict_output)
# predict_output = tf.reshape(predict_output, [predict_output_shape[0], predict_output_shape[1], predict_output_shape[2]])
predict_output = tf.transpose(tf.convert_to_tensor(predict_output), perm=[1, 0, 2])
predict_output = tf.contrib.layers.fully_connected(predict_output, vocabulary_size, reuse=True, scope="rnn/resEmbedding", activation_fn=None)
predict_output = tf.argmax(predict_output, axis=2)
# _, _, predict_output, _ = tf.while_loop(predict_loop_cond, predict_loop_body, [tf.convert_to_tensor([dictionary["<GO>"]], dtype=tf.int64), predict_hxz, tf.convert_to_tensor([[dictionary["<GO>"]]]), tf.constant(0)], shape_invariants=[tf.TensorShape([1]), tf.TensorShape([None, embedding_size + z_hidden_size]), tf.TensorShape([1, None]), tf.TensorShape([])])

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  # saver.restore(sess, "model")
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
  batch_q, batch_a, batch_q_len, batch_a_len, real_qa_batch_size = generate_qa_batch(
      qa_batch_size)
  for step in range(train_times):
    feed_dict = {q_placeholder: batch_q, a_placeholder: batch_a,
                                              q_sequence_length: batch_q_len, a_sequence_length: batch_a_len, real_batch_size_placeholder: real_qa_batch_size, current_step: step}
    l, _ = sess.run([loss, train], feed_dict=feed_dict)
    '''
    l, _, o= sess.run([loss, train,predict_output], feed_dict=feed_dict)
    print(l)
    print([reverse_dictionary[int(e)] for e in o[0]])
    '''
    if step % 100 == 0:
      print(str(step) + ':' + str(l))
      saver.save(sess, "model")
      o = sess.run([predict_output], feed_dict=feed_dict)
      print([reverse_dictionary[int(e)] for e in o[0][0]])
