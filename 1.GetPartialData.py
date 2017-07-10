import string
fin = open("data/train.txt", "r")
fout = open("data/segTrain.txt", "w")
for _ in range(10000):
    fout.write(fin.readline().translate(string.maketrans("", ""), string.punctuation))
