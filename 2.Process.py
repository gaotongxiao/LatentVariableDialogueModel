import tensorflow as tf
import collections
import numpy as np
import random
import re
import math
from reader import *

qa_batch_size = 128 # how many Q-A pairs to train at one batch
train_times = 2001
output_step = 100 # you can see loss and prediction every output_step
predict_or_train = True # False: Only use pre-trained model to predict True: Train
load_model = False # whether load pre-trained model before starting, only effective when train == True
save_model = False # whether save model every output_step, only effective when train == True
predict_question = "What are you doing" #should be a question without any punctuation
max_response_length = 20 # for prediction, define the maximum response length
predict_batch_size = 1 # for prediction, currently this program only support one question per batch
embedding_size = 512  # Dimension of the embedding vector.

q, a, ql, al = read_and_decode("data/train.tfrecords")
q_placeholder, a_placeholder, q_sequence_length, a_sequence_length= tf.train.shuffle_batch(
    [q, a, ql, al], batch_size=qa_batch_size, capacity=2000, min_after_dequeue=1000)

# reverse_dictionary: int -> word
# Note: UNK stands for unknown word that is not in the dictionary
reverse_dictionary = read_dictionary("data/dictionary.json")
################## temporarily use
dictionary = dict(zip(reverse_dictionary.values(), reverse_dictionary.keys()))
for k in dictionary.keys():
  dictionary[k] = int(dictionary[k])
###################
vocabulary_size = len(reverse_dictionary)
embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

#Now we get the word embedding vector, start building biRNN

current_step = tf.placeholder(tf.float32) # for annealing KL term
q_placeholder = tf.sparse_tensor_to_dense(q_placeholder, 0)
a_placeholder = tf.sparse_tensor_to_dense(a_placeholder, 0)
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
      q_fw_cell, q_bw_cell, q_embedded, sequence_length=q_sequence_length, dtype=tf.float32)
with tf.variable_scope("aRNN"):
  _, hy = tf.nn.bidirectional_dynamic_rnn(
      a_fw_cell, a_bw_cell, a_embedded, sequence_length=a_sequence_length, dtype=tf.float32)
hx = tf.concat([hx[0], hx[1]], 1)
hx = tf.contrib.layers.fully_connected(hx, embedding_size, activation_fn=tf.tanh, scope="hxFinalState")
hy = tf.concat([hy[0], hy[1]], 1)
hy = tf.contrib.layers.fully_connected(hy, embedding_size, activation_fn=tf.tanh)
hxhy = tf.concat([hx, hy], 1)

#Get mean and variance of latent variable z
z_hidden_size = 64
mean = tf.contrib.layers.fully_connected(hxhy, z_hidden_size, activation_fn=None)
sigma = tf.exp(tf.contrib.layers.fully_connected(hxhy, z_hidden_size, activation_fn=None) / 2.0) #div 2 -> sqrt
distribution = tf.random_normal([qa_batch_size, z_hidden_size])
z = tf.multiply(distribution, sigma) + mean  # dot multiply

#provide esstential information for training decoder
hxz = tf.concat([hx, z], 1)
hxz = tf.nn.dropout(hxz, 0.5)
with tf.variable_scope("decoderRNNCell"):
  decoder_cell = tf.nn.rnn_cell.GRUCell(embedding_size + z_hidden_size)
with tf.variable_scope("decoderRNN"):
  result, _ = tf.nn.dynamic_rnn(
      decoder_cell, a_embedded, sequence_length=a_sequence_length, initial_state=hxz)
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
predict_q_sequence_length = [len(e) for e in predict_questions_int]
with tf.variable_scope("predict_qRNN"):
  _, predict_hx = tf.nn.bidirectional_dynamic_rnn(
      q_fw_cell, q_bw_cell, predict_q_embedded, sequence_length=predict_q_sequence_length, dtype=tf.float32)
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
  threads = tf.train.start_queue_runners(sess)

  if predict_or_train == False or (predict_or_train == True and load_model == True):
    saver.restore(sess, "model")

  for step in range(train_times):
    feed_dict = {current_step: step}
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
        print([reverse_dictionary[str(e)] for e in o[0][0]])
