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

    # add the word 'START' to the beginning of every sentence and the word 'STOP' to the end of every sentence
    def add_start_stop_word(self):#todo check error
        for i in range(len(training_set)):
            self.bigram_training_set[i].insert(0, ('START', 'START'))
            self.bigram_training_set[i].append(('STOP', 'STOP'))

    # compute the count of every word and every tuples of words in the corpus sentences
    def count_words(self):
        word_tuple_count = {}
        word_count = {}
        for sent in self.bigram_training_set:
            for i in range(1, len(sent)):
                if sent[i][0] in word_count:
                    word_count[sent[i][0]] += 1
                else:
                    word_count[sent[i][0]] = 1
                if (sent[i][0], sent[i - 1][0]) in word_tuple_count:
                    word_tuple_count[(sent[i][0], sent[i - 1][0])] += 1
                else:
                    word_tuple_count[(sent[i][0], sent[i - 1][0])] = 1
        return word_tuple_count, word_count

    # compute the count of every tag and every tuples of tags in the corpus sentences
    def count_tags(self):
        tags_tuples_count = {}
        tags_count = {}
        for sent in self.bigram_training_set:
            for i in range(1, len(sent)):
                if sent[i][1] in tags_count:
                    tags_count[sent[i][1]] += 1
                else:
                    tags_count[sent[i][1]] = 1
                if (sent[i][1], sent[i - 1][1]) in tags_tuples_count:
                    tags_tuples_count[(sent[i][1], sent[i - 1][1])] += 1
                else:
                    tags_tuples_count[(sent[i][1], sent[i - 1][1])] = 1
        return tags_tuples_count, tags_count

    # find the emission probabilities of tuples of words
    def tuple_emission_prob(self, w1, w2, add_ones=False):
        tuples_count, word_count = self.count_words()
        if w2 in word_count and (w1, w2) in tuples_count:
            return tuples_count[(w1, w2)] / word_count[w2]
        else:
            return 0  # todo check option of add_ones
    def tuple_transition_prob(self,tag1,tag2):
        # todo implement

    def sent_prob(self,sent):
        prob=0
        for i in range(1,len(sent)):
            prob*=self.tuple_emission_prob(sent[i-1][0],sent[i][0])*self.tuple_transition_prob(sent[i-1][1],sent[i][1])
        return prob


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
