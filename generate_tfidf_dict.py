from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
import numpy as np

from nltk.corpus import stopwords
import csv, datetime, time, json
import pickle
QUESTION_PAIRS_FILE = 'keras_training_data.json'
QUESTION_PAIRS_FILE2 = 'testing_query.json'
MAX_NB_WORDS = 220000
tf1_file = 'tf1_dict.txt'
tf2_file = 'tf2_dict.txt'

idf_file =  'idf_dict.txt'

question1 = []
question2 = []
with open(QUESTION_PAIRS_FILE, encoding='utf-8') as jsondata:
    file = json.load(jsondata)
    flag = 0
    for row in file:
        if row['is_duplicate'] != 0 and row['is_duplicate'] != 1:
            pass
        else:
            question1.append(row['question1'])
            question2.append(row['question2'])
with open(QUESTION_PAIRS_FILE2, encoding='utf-8') as jsondata:
    file = json.load(jsondata)
    flag = 0
    for row in file:
        if row['is_duplicate'] != 0 and row['is_duplicate'] != 1:
            pass
        else:
            question1.append(row['question1'])
            question2.append(row['question2'])

print('Question pairs: %d' % len(question1))

# Build tokenized word index
questions = question1 + question2
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(questions)

#q = 'I have a apple and apple dont lie'
#q = text_to_word_sequence(q)
#print(q)
question1_word_sequences = tokenizer.texts_to_sequences(question1)
question2_word_sequences = tokenizer.texts_to_sequences(question2)
word_index = tokenizer.word_index
print("Words in index: %d" % len(word_index))
questions = []
questions1 = []
questions2 = []
for i in range(len(question1)):
    questions1.append(text_to_word_sequence(question1[i]))
    questions2.append(text_to_word_sequence(question2[i]))
    #like this ['I', 'have', 'a', 'apple', 'and', 'it', 'is', 'nice']

stopset = stopwords.words('english')
print(stopset)
for i in range(len(questions1)):
    for j in questions1[i]:
        if j in stopset:
            questions1[i].remove(j)

for i in range(len(questions2)):
    for j in questions2[i]:
        if j in stopset:
            questions2[i].remove(j)

tf1 = {}
for i in range(len(questions1)):
    if i % 5000 ==0 : print(i)
    tf1[i] = {}
    for j in questions1[i]:
        tf1[i][j] = 0
    for j in questions1[i]:
        tf1[i][j] += round((1.0/len(questions1[i])),5)

file1 = open(tf1_file,'wb')
pickle.dump(tf1,file1)

tf2 = {}
for i in range(len(questions2)):
    if i % 5000 ==0 : print(i)
    tf2[i] = {}
    for j in questions2[i]:
        tf2[i][j] = 0
    for j in questions2[i]:
        tf2[i][j] += round((1.0/len(questions2[i])),5)

file2 = open(tf2_file,'wb')
pickle.dump(tf2,file2)

idf = {}
for i in range(len(questions1)):
    for j in questions1[i]:
        idf[j] = 0
    for k in questions2[i]:
        idf[k] = 0

print(len(idf))

for i in range(len(questions1)):
    for j in questions1[i]:
        idf[j] += 1
    for j in questions2[i]:
        idf[j] += 1

for i in idf:
    idf[i] = np.log( len(questions1)/idf[i] )

file3 = open(idf_file,'wb')
pickle.dump(idf,file3)


        


