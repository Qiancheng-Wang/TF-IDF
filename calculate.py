
import numpy as np
import math
import csv, datetime, time, json
import pickle
QUESTION_PAIRS_FILE = 'testing_query.json'

tf1 = {}
tf2 = {}
idf = {}

tf1_file = 'tf1_dict.txt'
tf2_file = 'tf2_dict.txt'
idf_file =  'idf_dict.txt'

file1 = open(tf1_file,'rb')
tf1 = pickle.load(file1)

file2 = open(tf2_file,'rb')
tf2 = pickle.load(file2)

file3 = open(idf_file,'rb')
idf = pickle.load(file3)

def calculate(tf_1,tf_2):

    sum = 0
    for i in  tf_1:
        if i in tf_2:
            sum += tf_1[i]*tf_2[i]
        else:
            pass

    mean1 = 0
    for i in tf_1:
        mean1 += tf_1[i]*tf_1[i]
    mean1 = math.sqrt(mean1)

    mean2 = 0
    for i in tf_2:
        mean2 += tf_2[i] * tf_2[i]
    mean2 = math.sqrt(mean2)

    final = sum/(mean1*mean2)

    
    return round(final)



is_duplicate = []
with open('keras_training_data.json', encoding='utf-8') as jsondata:
    file = json.load(jsondata)
    flag = 0
    for row in file:
        if row['is_duplicate'] != 0 and row['is_duplicate'] != 1:
            pass
        else:
            is_duplicate.append(row['is_duplicate'])
with open(QUESTION_PAIRS_FILE, encoding='utf-8') as jsondata:
    file = json.load(jsondata)
    flag = 0
    for row in file:
        if row['is_duplicate'] != 0 and row['is_duplicate'] != 1:
            pass
        else:
            is_duplicate.append(row['is_duplicate'])

TP = 0
TN = 0
FP = 0
FN = 0

for i in range(0,450000):
    if i % 3000 == 0 : print(i)

    if calculate(tf1[i],tf2[i]) == 1 and is_duplicate[i-420000] == 1:
        TP += 1
    elif calculate(tf1[i],tf2[i]) == 0 and is_duplicate[i-420000] == 0:
        TN += 1
    elif calculate(tf1[i],tf2[i]) == 0 and is_duplicate[i-420000] == 1:
        FN += 1
    elif calculate(tf1[i],tf2[i]) == 1 and is_duplicate[i-420000] == 0:
        FP += 1

N = len(is_duplicate)
accuracy = (TP+TN)/N
precision =TP/(TP+FP)
recall = TP/(TP+FN)
f1 = (2*precision*recall)/(precision+recall)
print("Accuracy: ",accuracy)
print("Precision: ",precision)
print("Recall: ",recall)
print("F1 score: ",f1)
