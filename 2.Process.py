import tensorflow as tf
import collections
import numpy as np
import random
import re
import math

qa_batch_size = 128 # how many Q-A pairs to train at one batch
train_times = 2001
output_step = 100 # you can see loss and prediction every output_step
predict_or_train = True # False: Only use pre-trained model to predict True: Train
load_model = False # whether load pre-trained model before starting, only effective when train == True
save_model = True # whether save model every output_step, only effective when train == True
predict_question = "What are you doing" #should be a question without any punctuation
max_response_length = 20 # for prediction, define the maximum response length
predict_batch_size = 1 # for prediction, currently this program only support one question per batch

# Get word embedding vector
# questions and answers are a nested lists of words, with <GO> and <EOS> inserted (for RNN)
# vocalbulary is a list of all words in questions and answers (for embedding propose)
f = open("data/segTrain.txt", "r")
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

vocabulary_size = 5000 #define maximum vocalbulary size
vocabulary_size += 2  # for <GO> and <EOS>

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

data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)
# reversed_dictionary: int -> word
# dictionary: word -> int
# Note: UNK stands for unknown word that is not in the dictionary

#for testing
'''
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
'''
del vocabulary, count, data

batch_size = 128 # Training batch size
embedding_size = 512  # Dimension of the embedding vector.

embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

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
current_qa_index = 0 #for generate_qa_batch's use
qa_pairs_count = len(questions) 

# return:
# batch_q, batch_a: batch of lists of integer, it will pad sequences with 0(dictionary['UNK']) if that sequence length is shorter than the maximum in current batch 
# batch_q_len, batch_a_len: lists of shape [qa_batch_size, 1], standing for each question and answer's length 
# qa_batch_size: real batch shape, useful when the expected batch size is larger than real one
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


current_step = tf.placeholder(tf.float32) # for annealing KL term
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
hx = tf.contrib.layers.fully_connected(hx, embedding_size, activation_fn=tf.tanh, scope="hxFinalState")
hy = tf.concat([hy[0], hy[1]], 1)
hy = tf.contrib.layers.fully_connected(hy, embedding_size, activation_fn=tf.tanh)
hxhy = tf.concat([hx, hy], 1)

#Get mean and variance of latent variable z
z_hidden_size = 64
mean = tf.contrib.layers.fully_connected(hxhy, z_hidden_size, activation_fn=None)
sigma = tf.exp(tf.contrib.layers.fully_connected(hxhy, z_hidden_size, activation_fn=None) / 2.0) #div 2 -> sqrt
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

# function for calculating anneal KL term's weight
max_kl_weight = 1 # (0, 1]
def kl_weight_cal():
  return tf.cond(tf.less(current_step, train_times * 0.9), lambda: (max_kl_weight / (0.8 * train_times) * (current_step - 0.1 * train_times)), lambda: tf.cast(max_kl_weight, tf.float32))

#calculate loss function
square_sigma = tf.square(sigma)
square_mean = tf.square(mean)
a_target = tf.pad(a_placeholder[:, 1:], [[0, 0], [0, 1]]) # target answer
loss = 0.5 * tf.reduce_mean(1 + tf.log(square_sigma) - square_mean -
                           square_sigma)
kl_annealing_weight = tf.cond(tf.less(
    current_step, train_times / 10), lambda: tf.cast(0, tf.float32), kl_weight_cal);
loss *= kl_annealing_weight
kl_lower_bound = 0.0002 #Level of KL loss at which to stop optimizing for KL.
loss = tf.cond(tf.less(loss, kl_lower_bound), lambda: kl_lower_bound, lambda: loss) #avoid get NAN in loss function
loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=result,
                                                               labels=tf.one_hot(a_target, vocabulary_size, dtype=tf.float32)))
train = tf.train.AdamOptimizer().minimize(loss)

saver = tf.train.Saver()

#now let's predict

go_embedded = tf.nn.embedding_lookup(embeddings, tf.tile([dictionary["<GO>"]], [predict_batch_size]))
eos_embedded = tf.nn.embedding_lookup(embeddings, tf.tile([dictionary["<EOS>"]], [predict_batch_size]))

# for raw_rnn's use
# In prediction, we needs to feed cell's previous output to its next input
# However tensorflow lacks this implementation, so I just use raw_rnn to customize
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
    output_logits = tf.contrib.layers.fully_connected(previous_output, vocabulary_size, activation_fn=None, scope="resEmbedding", reuse=True)
    predict = tf.argmax(output_logits, axis=1)
    ret = tf.nn.embedding_lookup(embeddings, predict)
    ret = tf.reshape(ret, [predict_batch_size, embedding_size]) #just for skipping while_loop's shape checking
    return ret
  element_finished = (time >= max_response_length)
  finished = tf.reduce_all(element_finished)
  next_input = tf.cond(finished, lambda: eos_embedded, get_next_input)
  return (element_finished, next_input, previous_state, previous_output, None)

predict_questions = [predict_question.split()]
predict_questions_int = []
for question in predict_questions:
  words = []
  for word in question: 
    try:
      words.append(dictionary[word])
    except:
      words.append(dictionary["UNK"])
  predict_questions_int.append(words)
predict_q_embedded = tf.nn.embedding_lookup(embeddings, predict_questions_int)
predict_q_sequence_length = [[len(e)] for e in predict_questions_int]
with tf.variable_scope("predict_qRNN"):
  _, predict_hx = tf.nn.bidirectional_dynamic_rnn(
      q_fw_cell, q_bw_cell, predict_q_embedded, sequence_length=tf.squeeze(predict_q_sequence_length, 1), dtype=tf.float32)
predict_hx = tf.concat([predict_hx[0], predict_hx[1]], 1)
predict_hx = tf.contrib.layers.fully_connected(predict_hx, embedding_size, activation_fn=tf.tanh, scope="hxFinalState", reuse=True)
predict_z = tf.random_normal([1, z_hidden_size])
predict_hxz = tf.concat([predict_hx, predict_z], 1)
(predict_output, _, _) = tf.nn.raw_rnn(decoder_cell, loop_fn)
predict_output = predict_output.stack()
#now predict_output has shape [max_response_length, predict_batch_size, embedding_size+z_hidden_size]
predict_output = tf.transpose(tf.convert_to_tensor(predict_output), perm=[1, 0, 2])
predict_output = tf.contrib.layers.fully_connected(predict_output, vocabulary_size, reuse=True, scope="rnn/resEmbedding", activation_fn=None)
predict_output = tf.argmax(predict_output, axis=2)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  if predict_or_train == False or (predict_or_train == True and load_model == True):
    saver.restore(sess, "model")

  batch_q, batch_a, batch_q_len, batch_a_len, real_qa_batch_size = generate_qa_batch(
      qa_batch_size)
  for step in range(train_times):
    feed_dict = {q_placeholder: batch_q, a_placeholder: batch_a,
                                              q_sequence_length: batch_q_len, a_sequence_length: batch_a_len, real_batch_size_placeholder: real_qa_batch_size, current_step: step}
                                              # Only for prediction
    if predict_or_train == False:
      o = sess.run([predict_output], feed_dict=feed_dict)
      print([reverse_dictionary[int(e)] for e in o[0][0]])
    else:
      l, _ = sess.run([loss, train], feed_dict=feed_dict)
      if step % output_step == 0:
        print(str(step) + ':' + str(l))
        if save_model == True:
          saver.save(sess, "model")
        o = sess.run([predict_output], feed_dict=feed_dict)
        print([reverse_dictionary[int(e)] for e in o[0][0]])
