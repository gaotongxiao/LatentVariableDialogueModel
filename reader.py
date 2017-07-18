import tensorflow as tf

def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    
    _, example = reader.read(filename_queue)
    data = tf.parse_single_example(example, features={
        "question": tf.VarLenFeature(tf.int64),
        "answer": tf.VarLenFeature(tf.int64)
    })

    return data["question"], data["answer"] 

if __name__ == "__main__":
    q, a = read_and_decode("data/train.tfrecords")
    q_batch, a_batch = tf.train.shuffle_batch(
        [q, a], batch_size=2, capacity=2000, min_after_dequeue=1000)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    threads = tf.train.start_queue_runners(sess)
    print(sess.run([q_batch, a_batch]))
