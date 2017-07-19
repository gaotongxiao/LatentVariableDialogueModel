import tensorflow as tf
import collections
import numpy as np
import random
import re
import math
from reader import *
from build_graph import *

#common settings
model_path = "model/model"
dictionary_path = "data/dictionary.json"
embedding_size = 512  # Dimension of the embedding vector.
z_hidden_size = 64  # Get mean and variance of latent variable z

#settings for training
qa_batch_size = 64  # how many Q-A pairs to train at one batch
train_times = 200
output_step = 100  # you can see loss every output_step
# whether load pre-trained model before starting, only effective when train == True
load_model = False
save_model = True  # whether save model every output_step, only effective when train == True
save_step = train_times / 2  # program will save model every save_step
train_data_path = "data/train.tfrecords"
max_kl_weight = 1  # (0, 1]
kl_lower_bound = 0.0002  # Level of KL loss at which to stop optimizing for KL.

#settings for prediction
max_response_length = 20  # for prediction, define the maximum response length
# for prediction, currently this program only support one question per batch
predict_batch_size = 2
# for predction, determine how many times the program should answer each question
predict_times = 3
predict_output_path = "data/predict_output.txt"
valid_data_path = "data/valid.tfrecords"


def get_default_graph(vocabulary_size):
  training_graph = tf.Graph()
  with training_graph.as_default():
    q, a, ql, al = read_and_decode(train_data_path)
    q_placeholder, a_placeholder, q_sequence_length, a_sequence_length = tf.train.shuffle_batch(
        [q, a, ql, al], batch_size=qa_batch_size, capacity=qa_batch_size * 2, min_after_dequeue=qa_batch_size)

    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

    #Now we get the word embedding vector, start building biRNN
    current_step = tf.placeholder(tf.float32)  # for annealing KL term
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
    hx = tf.contrib.layers.fully_connected(
        hx, embedding_size, activation_fn=tf.tanh, scope="hxFinalState")
    hy = tf.concat([hy[0], hy[1]], 1)
    hy = tf.contrib.layers.fully_connected(
        hy, embedding_size, activation_fn=tf.tanh)
    hxhy = tf.concat([hx, hy], 1)

    mean = tf.contrib.layers.fully_connected(
        hxhy, z_hidden_size, activation_fn=None)
    sigma = tf.exp(tf.contrib.layers.fully_connected(
        hxhy, z_hidden_size, activation_fn=None) / 2.0)  # div 2 -> sqrt
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
    result = tf.contrib.layers.fully_connected(
        result, vocabulary_size, scope="rnn/resEmbedding", activation_fn=None)

    # function for calculating anneal KL term's weight
    def kl_weight_cal():
      return tf.cond(tf.less(current_step, train_times * 0.9), lambda: (max_kl_weight / (0.8 * train_times) * (current_step - 0.1 * train_times)), lambda: tf.cast(max_kl_weight, tf.float32))

    #calculate loss function
    square_sigma = tf.square(sigma)
    square_mean = tf.square(mean)
    a_target = tf.pad(a_placeholder[:, 1:], [[0, 0], [0, 1]])  # target answer
    loss = 0.5 * tf.reduce_mean(1 + tf.log(square_sigma) - square_mean -
                                square_sigma)
    kl_annealing_weight = tf.cond(tf.less(
        current_step, train_times / 10), lambda: tf.cast(0, tf.float32), kl_weight_cal);
    loss *= kl_annealing_weight
    loss = tf.cond(tf.less(loss, kl_lower_bound), lambda: kl_lower_bound,
                   lambda: loss)  # avoid get NAN in loss function
    loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=result,
                                                                   labels=tf.one_hot(a_target, vocabulary_size, dtype=tf.float32)))
    train = tf.train.AdamOptimizer().minimize(loss)
    tf.add_to_collection("current_step", current_step)
    tf.add_to_collection("loss", loss)
    tf.add_to_collection("train", train)
    tf.add_to_collection("embeddings", embeddings)
    tf.add_to_collection("q_cells", q_fw_cell)
    tf.add_to_collection("q_cells", q_bw_cell)
    tf.add_to_collection("decoder_cell", decoder_cell)
  return training_graph


def train(graph):
  with graph.as_default():
    saver = tf.train.Saver()
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      if load_model:
        saver.restore(sess, model_path)

      try:
        for step in range(train_times):
          feed_dict = {tf.get_collection("current_step")[0]: step}
          l, _ = sess.run([tf.get_collection("loss")[0],
                           tf.get_collection("train")[0]], feed_dict=feed_dict)
          if step % output_step == 0:
            print(str(step) + ':' + str(l))
          if save_model and step and step % save_step == 0:
            saver.save(sess, model_path)
            print("Model saved")
      except:
        print("Training aborted")
      finally:
        coord.request_stop()
      coord.join(threads)
      if save_model:
        saver.save(sess, model_path)
        print("model saved")
  return


def get_dictionary(path):
  # reverse_dictionary: int -> word
  # Note: UNK stands for unknown word that is not in the dictionary
  reverse_dictionary = read_dictionary(path)
  vocabulary_size = len(reverse_dictionary)
  return reverse_dictionary, vocabulary_size

def translate_int_to_string(int_list, reverse_dictionary):
  ret = ""
  for i in range(1, len(int_list) - 1):
    word = reverse_dictionary[str(int_list[i])]
    if word == "<EOS>":
      break
    ret += word + " "
  return ret

def predict(graph, reverse_dictionary, vocabulary_size):
  with graph.as_default():
    saver = tf.train.Saver()
    embeddings = tf.get_collection("embeddings")[0]
    p_q, p_a, p_ql, p_al = read_and_decode(valid_data_path)
    p_q_placeholder, p_a_placeholder, p_q_sequence_length, p_a_sequence_length = tf.train.shuffle_batch(
        [p_q, p_a, p_ql, p_al], batch_size=predict_batch_size, capacity=predict_batch_size * 2, min_after_dequeue=predict_batch_size)
    p_q_placeholder = tf.sparse_tensor_to_dense(p_q_placeholder)
    p_a_placeholder = tf.sparse_tensor_to_dense(p_a_placeholder)

    # 1:<GO> 2:<EOS>
    go_embedded = tf.nn.embedding_lookup(
        embeddings, tf.tile([1], [predict_batch_size * predict_times]))
    eos_embedded = tf.nn.embedding_lookup(
        embeddings, tf.tile([2], [predict_batch_size * predict_times]))

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
      element_finished = (0 >= max_response_length)
      return (element_finished, go_embedded, predict_hxz, None, None)

    def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
      def get_next_input():
        output_logits = tf.contrib.layers.fully_connected(
            previous_output, vocabulary_size, activation_fn=None, scope="resEmbedding", reuse=True)
        predict = tf.argmax(output_logits, axis=1)
        ret = tf.nn.embedding_lookup(embeddings, predict)
        # just for skipping while_loop's shape checking
        ret = tf.reshape(
            ret, [predict_batch_size * predict_times, embedding_size])
        return ret
      element_finished = (time >= max_response_length)
      finished = tf.reduce_all(element_finished)
      next_input = tf.cond(finished, lambda: eos_embedded, get_next_input)
      return (element_finished, next_input, previous_state, previous_output, None)

    predict_q_embedded = tf.nn.embedding_lookup(embeddings, p_q_placeholder)
    predict_q_embedded = tf.tile(predict_q_embedded, [predict_times, 1, 1])
    p_q_sequence_length = tf.tile(p_q_sequence_length, [predict_times])
    with tf.variable_scope("predict_qRNN"):
      _, predict_hx = tf.nn.bidirectional_dynamic_rnn(
          tf.get_collection("q_cells")[0], tf.get_collection("q_cells")[1], predict_q_embedded, sequence_length=p_q_sequence_length, dtype=tf.float32)
    predict_hx = tf.concat([predict_hx[0], predict_hx[1]], 1)
    predict_hx = tf.contrib.layers.fully_connected(
        predict_hx, embedding_size, activation_fn=tf.tanh, scope="hxFinalState", reuse=True)
    predict_z = tf.random_normal(
        [predict_batch_size * predict_times, z_hidden_size])
    predict_hxz = tf.concat([predict_hx, predict_z], 1)
    (predict_output, _, _) = tf.nn.raw_rnn(
        tf.get_collection("decoder_cell")[0], loop_fn)
    predict_output = predict_output.stack()
    #now predict_output has shape [max_response_length, predict_batch_size, embedding_size+z_hidden_size]
    predict_output = tf.transpose(
        tf.convert_to_tensor(predict_output), perm=[1, 0, 2])
    predict_output = tf.contrib.layers.fully_connected(
        predict_output, vocabulary_size, activation_fn=None, scope="rnn/resEmbedding", reuse=True)
    predict_output = tf.arg_max(predict_output, 2)

    with tf.Session() as sess:
      saver.restore(sess, model_path)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      questions, answers, predict_answer = sess.run([p_q_placeholder, p_a_placeholder, predict_output])
      coord.request_stop()
      coord.join(threads)
      print(translate_int_to_string(answers[0], reverse_dictionary))
      exit()
      ans = o[0].tolist()
      for a in ans:
        print([reverse_dictionary[str(w)] for w in a])
  return


if __name__ == "__main__":
  reverse_dictionary, vocabulary_size = get_dictionary(dictionary_path)
  g = get_default_graph(vocabulary_size)
  predict(g, reverse_dictionary, vocabulary_size)
