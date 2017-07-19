import tensorflow as tf
import json

def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    
    _, example = reader.read(filename_queue)
    data = tf.parse_single_example(example, features={
        "question": tf.VarLenFeature(tf.int64),
        "answer": tf.VarLenFeature(tf.int64),
        "question_length": tf.FixedLenFeature([], tf.int64),
        "answer_length": tf.FixedLenFeature([], tf.int64)
    })

    return data["question"], data["answer"], data["question_length"], data["answer_length"]

def read_dictionary(filename):
    return json.loads(open(filename, "r").readline())

if __name__ == "__main__":
    q, a, ql, al = read_and_decode("data/valid.tfrecords")
    q_batch, a_batch, ql_batch, al_batch = tf.train.shuffle_batch(
        [q, a, ql, al], batch_size=2, capacity=2000, min_after_dequeue=1000)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    threads = tf.train.start_queue_runners(sess)
    print(sess.run([q_batch, a_batch, ql_batch, al_batch]))
