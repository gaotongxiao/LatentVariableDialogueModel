import tensorflow as tf
import string
import json
import collections

# convert opensubtitle data into TFRecord, and save the dictionary (int -> word) to json file

origin_data_path = "data/train.txt"
training_data_save_path = "data/train.tfrecords"
# int -> word dictionary
dictionary_save_path = "data/dictionary.json"
# determine maximum question-answer pairs to convert & save, use -1 to process all pairs
num_pairs = -1
# how many words should be added into dictionary
num_words = 30000
# larger step speeds up the process but needs larger memory
count_word_step = 2000000

train_file = open(origin_data_path, "r")
dictionary_file = open(dictionary_save_path, "w")
writer = tf.python_io.TFRecordWriter(training_data_save_path)

linecount = 0
if num_pairs == -1:
    for linecount, _ in enumerate(train_file):
        pass
    linecount += 1
else:
    linecount = num_pairs
train_file.seek(0)

words = []
texts = []

#process and save QA pairs
counter = collections.Counter()
for i in range(linecount):
    try:
        line = train_file.readline().translate(string.maketrans("", ""), string.punctuation)
        if line == "":
            break
        line = line.strip("\r\n").split("\t")
        question = "<GO> " + line[0] + " <EOS>"
        answer = "<GO> " + line[1] + " <EOS>"
        question = question.split()
        answer = answer.split()
        '''
        '''
        words += question + answer
        # texts.append([question, answer])
        if i % count_word_step == 0 and i:
            print(i)
            counter += collections.Counter(dict(collections.Counter(words).most_common(num_words)))
            words = []
    except:
        break
counter += collections.Counter(dict(collections.Counter(words).most_common(num_words)))
train_file.close()
exit()

#create dictionary
count = [['UNK', -1]]
count.extend(collections.Counter(words).most_common(num_words - 1 + 2)) # extra 2 places for <GO> and <EOS>
del words
dictionary = dict()
for word, _ in count:
    dictionary[word] = len(dictionary)
reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
dictionary_file.write(json.dumps(reversed_dictionary))
del reversed_dictionary

for [question, answer] in texts:
    question_int = []
    answer_int = []
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
    writer.write(example.SerializeToString())

writer.close()
