import tensorflow as tf
import string
import json
import collections

# convert opensubtitle data into TFRecord, and save the dictionary (int -> word) to json file

train_data_load_path = "data/train.txt"
valid_data_load_path = "data/valid.txt"
train_data_save_path = "data/train.tfrecords"
valid_data_save_path = "data/valid.tfrecords"
# int -> word dictionary
dictionary_save_path = "data/dictionary.json"
# (for training) determine maximum question-answer pairs to convert & save, use -1 to process all pairs
train_num_pairs = -1
# how many words should be added into dictionary
num_words = 30000
# larger batch size speeds up the process but needs larger memory
process_batch_size = 3000000
# (for validating) determine maximum question-answer pairs to convert & save, use -1 to process all pairs
valid_num_pairs = -1

train_file = open(train_data_load_path, "r")
valid_file = open(valid_data_load_path, "r")
dictionary_file = open(dictionary_save_path, "w")

train_linecount = 0
if train_num_pairs == -1:
    for train_linecount, _ in enumerate(train_file):
        pass
    train_linecount += 1
else:
    train_linecount = train_num_pairs

valid_linecount = 0
if valid_num_pairs == -1:
    for valid_linecount, _ in enumerate(valid_file):
        pass
    valid_linecount += 1
else:
    valid_linecount = valid_num_pairs
valid_file.close()

words = []

#process and save QA pairs
print("Creating dictionary...")
train_file.seek(0)
counter = collections.Counter()
for i in range(train_linecount):
    line = train_file.readline().translate(
        string.maketrans("", ""), string.punctuation)
    if line == "":
        break
    words += line.split()
    # texts.append([question, answer])
    if i % process_batch_size == 0 and i:
        print(str(i) + "/" + str(train_linecount))
        counter.update(
            dict(collections.Counter(words).most_common(num_words)))
        words = []
counter.update(dict(collections.Counter(words).most_common(num_words)))
del words
train_file.close()

count = [['UNK', -1], ['<GO>', -1], ['<EOS>', -1]]
count.extend(counter.most_common(num_words - 1))  # minus 1 for UNK
del counter
dictionary = dict()
for word, _ in count:
    dictionary[word] = len(dictionary)
reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
dictionary_file.write(json.dumps(reversed_dictionary))
del reversed_dictionary


def translate_file(load_path, save_path, dictionary, num_lines):
    load_file = open(load_path, "r")
    save_file = tf.python_io.TFRecordWriter(save_path)
    for i in range(num_lines):
        if i % process_batch_size == 0 and i:
            print(str(i) + "/" + str(num_lines))
        question_int = []
        answer_int = []
        line = load_file.readline().translate(
            string.maketrans("", ""), string.punctuation)
        if line == "":
            break
        line = line.strip("\r\n").split("\t")
        question = ["<GO>"] + line[0].split() + ["<EOS>"]
        answer = ["<GO>"] + line[1].split() + ["<EOS>"]
        for word in question:
            try:
                question_int.append(dictionary[word])
            except:
                question_int.append(dictionary["UNK"])
        for word in answer:
            try:
                answer_int.append(dictionary[word])
            except:
                answer_int.append(dictionary["UNK"])
        example = tf.train.Example(features=tf.train.Features(feature={
            "question": tf.train.Feature(int64_list=tf.train.Int64List(value=question_int)),
            "answer": tf.train.Feature(int64_list=tf.train.Int64List(value=answer_int)),
            "question_length": tf.train.Feature(int64_list=tf.train.Int64List(value=[len(question_int)])),
            "answer_length": tf.train.Feature(int64_list=tf.train.Int64List(value=[len(answer_int)]))}))
        save_file.write(example.SerializeToString())
    load_file.close()
    save_file.close()


print("Creating training data...")
translate_file(train_data_load_path, train_data_save_path,
               dictionary, train_linecount)
print("Creating validating data...")
translate_file(valid_data_load_path, valid_data_save_path,
               dictionary, valid_linecount)
