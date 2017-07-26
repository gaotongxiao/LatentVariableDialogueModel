#the test result format is like: (when predict_times = 2)
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

predict_times = int(pred_file.readline())
print("predict_times: " + str(predict_times))
block_size = predict_times + 6
block_number = 0 #number of blocks(questions)

question = []
answer = []
prediction = []

#read file
while(pred_file.readline() == "Q:\r\n"):
	question.append(pred_file.readline())
	if pred_file.readline() != "A:\r\n":
		print("file format error(A:)")
	answer.append(pred_file.readline())
	if pred_file.readline() != "P:\r\n":
		print("file format error(P:)")
	current_prediction = []
	for i in range(predict_times):
		current_prediction.append(pred_file.readline())
	prediction.append(current_prediction)
	if pred_file.readline() != "------\r\n":
		print("file format error(------)")
	block_number += 1


#calculation of Zipf parameter:
words_to_use = 10 #use the top** highest frequency words to calculate zipf parameter
#frequency dictionary:
freq_dict = {}
for i in range(block_number):
	for j in range(predict_times):
		words = prediction[i][j].split()
		for word in words:
			try:
				freq_dict[word] += 1
			except:
				freq_dict[word] = 1
#a list of words sorted by freq
sorted_freq = sorted(freq_dict.values(), reverse = True)
print("Top Frequences Used:")
print(sorted_freq)
use_freq = sorted_freq[:words_to_use]
log_use_freq = [math.log(i) for i in use_freq]
log_index = [math.log(i) for i in range(1,words_to_use+1)]
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
	for j in range(predict_times):
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
unique_ratio = float(unique_count)/(predict_times*block_number)
print("unique_ratio: " + str(unique_ratio))

