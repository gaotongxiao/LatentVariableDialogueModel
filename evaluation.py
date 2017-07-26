#the test result format is like: (when predict_qa_size = 2)
#2
#Q:
#<Q>
#A:
#<A>
#P:
#<p1>
#<p2>
#------
#Q:
#<Q>

import string
import collections
import operator
import math
import numpy as np

predict_output_path = "data/predict_output.txt"
pred_file = open(predict_output_path, "r")
predict_stat_path = "data/stat.txt"
stat_file = open(predict_stat_path, "w")

predict_qa_size = int(pred_file.readline())
print("Predict_qa_size: " + str(predict_qa_size))
block_size = predict_qa_size + 6
block_number = 0 #number of blocks(questions)

question = []
answer = []
prediction = []

#read file
while(pred_file.readline() == "Q:\n"):
	question.append(pred_file.readline())
	if pred_file.readline() != "A:\n":
		print("file format error(A:)")
	answer.append(pred_file.readline())
	if pred_file.readline() != "P:\n":
		print("file format error(P:)")
	current_prediction = []
	for i in range(predict_qa_size):
		current_prediction.append(pred_file.readline())
	prediction.append(current_prediction)
	if pred_file.readline() != "------\n":
		print("file format error(------)")
	block_number += 1


#calculation of Zipf parameter:
words_to_use = 8 #use the top** highest frequency words to calculate zipf parameter
#frequency dictionary:
freq_dict = {}
for i in range(block_number):
	for j in range(predict_qa_size):
		words = prediction[i][j].split()
		for word in words:
			try:
				freq_dict[word] += 1
			except:
				freq_dict[word] = 1
#a list of words sorted by freq
print(freq_dict)
sorted_freq = sorted(freq_dict.values(), reverse = True)
print(sorted_freq)
use_freq = sorted_freq[:words_to_use]
log_use_freq = [math.log(i) for i in use_freq]
log_index = [math.log(i) for i in range(1,words_to_use+1)]
print(len(log_use_freq))#DEBUG
print(len(log_index))#DEBUG
#do linear regression
k,b = np.polyfit(log_index, log_use_freq, 1) #y = kx+b where x = log_index, y = log_use_freq
zipf_parameter = -k #here we estimate zipf parameter using linear regression under log scale of the PMF of the zipf distribution
print("zipf_parameter: " + str(zipf_parameter))

#we count the unique sentence with scope of each question
unique_count = 0
for i in range(block_number):
	unique_adder = 0 #the number of unique sentences of this question
	unique_sentences = []
	#update unique_adder in this for loop
	for j in range(predict_qa_size):
		sentence = prediction[i][j]
		duplicate = False
		for prev_sentence in unique_sentences:
			if sentence == prev_sentence:
				duplicate = True
				break
		if duplicate == False:
			unique_sentences.append(sentence)
			unique_adder +=1
	unique_count += unique_adder
unique_ratio = float(unique_count)/(predict_qa_size*block_number)
print("unique_ratio: " + str(unique_ratio))

