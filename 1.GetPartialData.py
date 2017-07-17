import string

#save 10000 lines Q-A data without punctuation into data/segTrain.txt
fin = open("data/train.txt", "r")
fout = open("data/segTrain.txt", "w")
for _ in range(10000):
    fout.write(fin.readline().translate(string.maketrans("", ""), string.punctuation))
