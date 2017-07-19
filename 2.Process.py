import tensorflow as tf
import collections
import numpy as np
import random
import re
import math
from reader import *

qa_batch_size = 64 # how many Q-A pairs to train at one batch
train_times = 2001
output_step = 100 # you can see loss and prediction every output_step
predict_or_train = False # False: Only use pre-trained model to predict True: Train
load_model = True # whether load pre-trained model before starting, only effective when train == True
save_model = True # whether save model every output_step, only effective when train == True
embedding_size = 512  # Dimension of the embedding vector.
train_data_path = "data/train.tfrecords"

#now let's predict


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  if predict_or_train == False or (predict_or_train == True and load_model == True):
    saver.restore(sess, "model")

  try:
    for step in range(train_times):
      feed_dict = {current_step: step}
      if predict_or_train == False:
        o = sess.run([predict_output], feed_dict=feed_dict)
        ans = o[0].tolist()
        for a in ans:
          print([reverse_dictionary[str(w)] for w in a])
        exit()
        print([reverse_dictionary[int(e)] for e in o[0][0]])
      else:
        l, _ = sess.run([loss, train], feed_dict=feed_dict)
        if step % output_step == 0:
          print(str(step) + ':' + str(l))
          if save_model == True:
            saver.save(sess, "model")
          o = sess.run([predict_output], feed_dict=feed_dict)
          print([reverse_dictionary[str(e)] for e in o[0][0]])
  except:
    print("Training aborted")
  finally:
    coord.request_stop()
  coord.join(threads)
