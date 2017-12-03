import nltk
import numpy as np
from nltk.corpus import brown
import numpy as np
import copy
from functools import reduce
import collections
from multiprocessing import Pool

STOP = 'STOP'
START = 'START'
USE_NN_IN_BIGRAM = True

data = brown.tagged_sents(categories='news')
training_set = data[0:int(len(data) * 0.9)]
test_set = []
tmp = data[int(len(data)*0.9):len(data)]
for sen in tmp:
    test_set.append([(START,START)] + sen + [(STOP,STOP)])


def get_all_possible_tags_and_words():
    all_tags = set()
    all_words = set()
    for sen in training_set:
        for tagged_word in sen:
            all_words.add(tagged_word[0])
            all_tags.add(tagged_word[1])
    return all_words, all_tags


class ngram:
    # dict(touple of (word,tag), counter)
    counters = {}

    # dict(touple of words, dict(tag, probability))
    probabilities = {}

    # sets of all possible words and tags
    all_words, all_tags = get_all_possible_tags_and_words()

    # count the tagged sentences into 'counters', implement in the derived classes
    def count(self, tagged_sents):
        print('not implemented!')
        return

    # calculate probabilities based on the counters dictionary, generic - implement here
    def calculate_probabilities(self):
        print('TBD')
        return

        # get tagged sentences and dict(word, ps_word),

    # iterate the sentences, and replace all the words appear in the dict with there corresponding pseudo word.
    def replace_pseudowords(self, tagged_sents, pseudowords):
        print('TBD')

    # iterate over all counters, add one to all of them.
    def add_one(self):
        keys = self.counters.keys()
        for key in keys:
            self.counters[key] = self.counters[key] + 1

    # implement in the derived
    def tag_sentence(self, sent):
        print('not implemented!')

    # return number of mistakes
    def score_sentence(self, tagged_sent):
        mistakes_counter = 0
        sentence = []
        for tagged_word in tagged_sent:
            sentence.append(tagged_word[0])
        # tag the sentence
        tagged_sents_our = self.tag_sentence(np.array(sentence, dtype='O'))
        if len(tagged_sents_our) != len(tagged_sent):
            print('ho no!')
        # check our tagging and update mistake counter
        for i in range(1, len(tagged_sents_our)-1):
            if tagged_sents_our[i][1] != tagged_sent[i][1]:
                mistakes_counter = mistakes_counter + 1
        return mistakes_counter

    # get sentences, tag them using tag_sentence(), compare to original, compute and return the accuarcy
    def test(self, tagged_sents):
        total_counter = 0
        a = list(map(self.score_sentence, tagged_sents))
        mistakes_counter = np.sum(a)
        for s in tagged_sents:
            total_counter-=2
            for w in s:
                total_counter+= 1
        return 1 - float(mistakes_counter) / total_counter

        # the full training method, implement in derived

    def train(self, tagged_sents):
        print('not imlemented!')

    # the full training method, implement in derived
    def train(self, tagged_sents):
        print('not imlemented!')


class unigram(ngram):
    words_highest_tag = {}

    def count(self, tagged_sents):
        for tag in self.all_tags:
            for word in self.all_words:
                self.counters[(word, tag)] = 0

        for sentence in tagged_sents:
            for tagged_word in sentence:
                self.counters[(tagged_word[0], tagged_word[1])] = self.counters[tagged_word] + 1
        for word in self.all_words:
            tags_counts = {}
            for tag in (self.all_tags):
                tags_counts[tag] = self.counters[(word, tag)]
            self.words_highest_tag[word] = max(tags_counts, key=tags_counts.get)

        return

    def train(self, tagged_sents):
        self.count(tagged_sents)

    def tag_sentence(self, sent):
        tagged_sentece = np.zeros(np.shape(sent), dtype='O')
        for i in range(np.shape(sent)[0]):
            if sent[i] in self.words_highest_tag:
                tagged_sentece[i] = (sent[i], self.words_highest_tag[sent[i]])
            else:
                tagged_sentece[i] = (sent[i], 'NN')
        return tagged_sentece


class bigram(ngram):
    add_one_flag = False
    bigram_training_set = []
    words_tags_count = collections.defaultdict(lambda: collections.defaultdict(int))
    tags_tuples_count = collections.defaultdict(lambda: collections.defaultdict(int))
    tags_count = collections.defaultdict(int)
    # we are smoothing using add-one only the transition!
    tags_count_transition = collections.defaultdict(int)
    all_tags_vec = np.array(1)

    def __init__(self, add_one_=False):
        # add the word 'START' to the beginning of every sentence and the word 'STOP' to the end of every sentence
        for sent in training_set:
            self.bigram_training_set.append([(START, START)] + sent + [(STOP, STOP)])
        ngram.all_tags.add(START)
        ngram.all_tags.add(STOP)
        ngram.all_words.add(STOP)
        sorted_tag_list = list(ngram.all_tags)
        sorted_tag_list.sort()
        self.all_tags_vec = np.array(sorted_tag_list).reshape(len(ngram.all_tags), 1)
        if add_one_:
            self.tags_tuples_count = collections.defaultdict(lambda: collections.defaultdict(lambda: 1))
            self.tags_count_transition = collections.defaultdict(lambda: len(self.all_tags))

        self.count_words_and_tags()

        # if add_one_:
            # self.add_one()
        self.create_transition_matrix()

    def add_one(self):
        for word in self.all_words:
            for tag in self.all_tags:
                self.words_tags_count[word][tag] += 1
        for tag1 in self.all_tags:
            self.tags_count[tag1] += 1
            for tag2 in self.all_tags:
                self.tags_tuples_count[tag1][tag2] += 1

    # compute the count of every tag and every tuples of tags and every tuples of words in the corpus sentences
    def count_words_and_tags(self):

        for sent in self.bigram_training_set:
            self.tags_count[sent[0][0]] += 1
            self.tags_count_transition[sent[0][0]] += 1
            for i in range(1, len(sent)):
                self.words_tags_count[sent[i][0]][sent[i][1]] += 1
                self.tags_count[sent[i][1]] += 1
                self.tags_count_transition[sent[i][1]] += 1
                self.tags_tuples_count[sent[i - 1][1]][sent[i][1]] += 1

    # use after training (counting)
    def create_transition_matrix(self):
        self.trans_prob_mat = 0 - np.ones(shape=(self.all_tags_vec.shape[0], self.all_tags_vec.shape[0]))
        for i in range(0, self.all_tags_vec.shape[0]):
            for j in range(0, self.all_tags_vec.shape[0]):
                self.trans_prob_mat[i, j] = self.tuple_transition_prob(self.all_tags_vec[i, 0], self.all_tags_vec[j, 0])
        self.trans_prob_mat = self.trans_prob_mat.transpose()

    # find the emission probability of word and tag
    def tuple_emission_prob(self, w, t, add_ones=False):
        if self.tags_count[t] == 0:
            return 0
        return self.words_tags_count[w][t] / self.tags_count[t]

    # find the transition probability of word and tag
    def tuple_transition_prob(self, tag1, tag2):
        if self.tags_count_transition[tag1] == 0:
            print('tag count is zero, tag=', tag1)
            return 0
        return self.tags_tuples_count[tag1][tag2] / self.tags_count_transition[tag1]

    # find the probability of a sentence according to MLE
    def sent_prob(self, sent):
        prob = 1
        for i in range(1, len(sent)):
            prob *= self.tuple_emission_prob(sent[i][0], sent[i][1]) * self.tuple_transition_prob(sent[i - 1][1],
                                                                                                  sent[i][1])
        return prob

    def viterbi_recrus(self, previous_state, current_word,previous_path):
        emission_vec = np.apply_along_axis(lambda tag: self.tuple_emission_prob(current_word, tag[0]),1,self.all_tags_vec)
        if np.sum(emission_vec)==0 and USE_NN_IN_BIGRAM:
            NN_index = np.where(self.all_tags_vec == "NN")[0][0]
            emission_vec[NN_index] = 1
        temp_mult_res = np.multiply(previous_state, self.trans_prob_mat)

        max_prob=np.max(temp_mult_res,axis=1).reshape(len(self.all_tags),1)
        max_prob_idx=np.argmax(temp_mult_res,axis=1).reshape(len(self.all_tags),1)
        cur_state=np.multiply(max_prob,emission_vec.reshape(len(self.all_tags),1))
        cur_path=previous_path[max_prob_idx, 0]
        for i in range(cur_path.shape[0]):
            cur_path[i,0] = cur_path[i,0] + [self.all_tags_vec[i][0]]
        i = 1

        return cur_state,cur_path

    def viterbi(self, sent):
        start_idx = np.where(self.all_tags_vec == START)[0][0]
        viterbi_table = np.full((len(self.all_tags), len(sent)), -1.0)
        path_vecor = np.empty((len(self.all_tags), 1), dtype='O')
        for i in range(len(self.all_tags)):
            path_vecor[i,0]=[]
        path_vecor[start_idx,0].append(START)
        viterbi_table[:, 0] = 0
        viterbi_table[start_idx][0] = 1
        for i in range(1,len(sent)):
            cur_state,cur_path=self.viterbi_recrus(viterbi_table[:,i-1],sent[i],path_vecor)
            viterbi_table[:, i]=cur_state[:, 0]
            path_vecor=cur_path
        final_path_idx=np.argmax(viterbi_table[:, len(sent)-1])
        return path_vecor[final_path_idx]

    def tag_sentence(self, sent):
        ret=self.viterbi(sent)[0]
        ret_list = []
        for i in range(len(sent)):
            ret_list.append((sent[i], ret[i]))
        return ret_list



model_bi = bigram(True)
print('accuarcy = ', model_bi.test(test_set))
