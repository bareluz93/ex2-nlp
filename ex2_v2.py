import nltk
import numpy as np
from nltk.corpus import brown
import numpy as np
import copy
from functools import reduce
from collections import defaultdict


STOP = 'STOP'

START = 'START'

data = brown.tagged_sents(categories='news')
training_set = []
tmp = data[0:int(len(data) * 0.9)]
for sent in tmp:
    training_set.append([(START, START)] + sent + [(STOP, STOP)])
test_set = []
tmp = data[int(len(data) * 0.9):len(data)]
for sent in tmp:
    test_set.append([(START, START)] + sent + [(STOP, STOP)])

def get_all_possible_tags_and_words():
    all_tags = set()
    all_words = set()
    for sen in training_set:
        for tagged_word in sen:
            all_words.add(tagged_word[0])
            all_tags.add(tagged_word[1])
    return all_words, all_tags

all_words, all_tags = get_all_possible_tags_and_words()

#counting!
words_count = defaultdict(int)
word_tag_count = defaultdict(int)
tag_tag_count = defaultdict(int)
for sent in training_set:
    words_count[sent[0][0]] +=1
    word_tag_count[(sent[0][0], sent[0][1])] +=1
    for i in range(1, len(sent)):
        words_count[sent[i][0]] +=1
        word_tag_count[(sent[i-1][1], sent[i][1])] +=1
        word_tag_count[(sent[i][0], sent[i][1])] +=1





# {t2, Pr(t2|t1)}
transition_probability = {}
for t2 in all_tags:
    for t1 in all_tags:
        transition_probability['t2'] =