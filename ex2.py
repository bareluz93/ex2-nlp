import nltk
import numpy as np
from nltk.corpus import brown
import numpy as np
import copy
from functools import reduce

data = brown.tagged_sents(categories='news')
training_set = data[0:int(len(data) * 0.9)]


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

    # get sentences, tag them using tag_sentence(), compare to original, compute and return the accuarcy
    def test(self, tagged_sents):
        total_counter = 0
        mistakes_counter = 0
        for tagged_sentence in tagged_sents:
            # create new, untagged, sentence
            sentence = []
            for tagged_word in tagged_sentence:
                sentence.append(tagged_word[0])
            # tag the sentence
            tagged_sents_our = self.tag_sentence(np.array(sentence, dtype='O'))
            if len(tagged_sents_our) != len(tagged_sentence):
                print('ho no!')
            # check our tagging and update mistake counter
            for i in range(0, len(tagged_sents_our)):
                total_counter = total_counter + 1
                if tagged_sents_our[i][1] != tagged_sentence[i][1]:
                    mistakes_counter = mistakes_counter + 1

        return 1 - float(mistakes_counter)/total_counter

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
            if (sent[i] in self.words_highest_tag):
                tagged_sentece[i] = (sent[i], self.words_highest_tag[sent[i]])
            else:
                tagged_sentece[i] = (sent[i], 'NN')
        return tagged_sentece


class bigram(ngram):
    bigram_training_set = training_set  # todo copy
    words_tags_count = collections.defaultdict(lambda: collections.defaultdict(int))
    tags_tuples_count = collections.defaultdict(lambda: collections.defaultdict(int))
    tags_count = collections.defaultdict(int)

    # add the word 'START' to the beginning of every sentence and the word 'STOP' to the end of every sentence
    def add_start_stop_word(self):#todo check error
        for i in range(len(training_set)):
            self.bigram_training_set[i].insert(0, ('START', 'START'))
            self.bigram_training_set[i].append(('STOP', 'STOP'))

    def add_one(self):
          for word in self.all_words:
              for tag in self.all_tags:
                  self.words_tags_count[word][tag]+=1
          for tag1 in self.all_tags:
              self.tags_count[tag1] += 1
              for tag2 in self.all_tags:
                  self.tags_tuples_count[tag1][tag2] += 1



    # compute the count of every tag and every tuples of tags and every tuples of words in the corpus sentences
    def count_words_and_tags(self):

        for sent in self.bigram_training_set:
            for i in range(1, len(sent)):
                self.words_tags_count[sent[i][0]][sent[i][1]] += 1
                self.tags_count[sent[i][1]] += 1
                self.tags_tuples_count[sent[i][1]][sent[i - 1][1]] += 1

    # find the emission probabilitiy of word and tag
    def tuple_emission_prob(self, w, t, add_ones=False):
        return self.words_tags_count[w][t] / self.tags_count[t]

    # def tuple_transition_prob(self,tag1,tag2):
    #     # todo implement

    def sent_prob(self,sent):
        prob=0
        for i in range(1,len(sent)):
            prob*=self.tuple_emission_prob(sent[i][0],sent[i][1])*self.tuple_transition_prob(sent[i-1][1],sent[i][1])
        return prob
    
        
    def calc_trans_prob(self,tagged_sents):
        all_words, all_tags = get_all_possible_tags_and_words()
        all_tags.add('START')
        all_tags.add('END')

        trans_count = {}
        tags_count = {}
        for t1 in all_tags:
            tags_count[t1] = 0
            for t2 in all_tags:
                trans_count[(t1, t2)] = 0

        for sen in tagged_sents:
            sen = sen.copy()
            sen = [('START', 'START')] + sen + [('END', 'END')]
            t0 = sen[0][1]
            tags_count[t0] = tags_count[t0] + 1
            for i in range(1, len(sen)):
                word = sen[i]
                t1 = word[1]
                tags_count[t1] = tags_count[t1] + 1
                trans_count[(t0,t1)] = trans_count[(t0,t1)] + 1
                t0 = t1

        trans_prob = {}
        for t1 in all_tags:
            trans_prob[('START', t1)] = 0
            trans_prob[(t1, 'END')] = 0
            for t2 in all_tags:
                trans_prob[(t1, t2)] = 0

        for t1 in all_tags:
            for t2 in all_tags:
                trans_prob[(t1, t2)] = trans_count[(t1, t2)] / tags_count[t1]
        self.trans_prob = trans_prob


    # def count(self, tagged_sents):
    #
    #     single_word_dict = {}
    #     for tag in self.all_tags:
    #         single_word_dict[tag] = 0
    #
    #     for word1 in self.all_words:
    #         for word2 in self.all_words:
    #             self.counters[(word1, word2)] = single_word_dict.copy()
    #     # todo add at any sentencs begining the * word, add * also in the counters
    #     # todo update counters
    #     return


model_bi = bigram()
tuples_count, words_count = model_bi.count_words()
# print('...........tuples_count.........')
# print(tuples_count)
print('...........words_count.........')
print(words_count['STOP'])
