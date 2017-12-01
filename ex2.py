import nltk
import numpy as np
from nltk.corpus import brown
import numpy as np

data = brown.tagged_sents(categories='news')
training_set = data[0:int(len(data) * 0.1)]


class ngram:
    # dict(touple of (word,tag), counter)
    counters = {}

    # dict(touple of words, dict(tag, probability))
    probabilities = {}

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
            print(tagged_sentence)
            # create new, untagged, sentence
            sentence = []
            for tagged_word in tagged_sentence:
                sentence.append(tagged_word[0])
            # tag the sentence
            tagged_sents_our = self.tag_sentence(np.array(sentence))
            # check our tagging and update mistake counter
            for tagged_word_our, tagged_word_orig in zip(tagged_sents_our, tagged_sents):
                total_counter = total_counter + 1
                if tagged_word_our[1] != tagged_word_orig[1]:
                    mistakes_counter = mistakes_counter + 1
        print(total_counter)
        print(mistakes_counter)
        print('total score is ', mistakes_counter/total_counter)



    # the full training method, implement in derived
    def train(self, tagged_sents):
        print('not imlemented!')


def get_all_possible_tags():
    all_tags = set()
    all_words = set()
    for sen in training_set:
        for tagged_word in sen:
            all_words.add(tagged_word[0])
            all_tags.add(tagged_word[1])
    return all_words, all_tags


class unigram(ngram):

    words_highest_tag = {}

    def count(self, tagged_sents):
        all_words, all_tags = get_all_possible_tags()
        for tag in all_tags:
            for word in  all_words:
                self.counters[(word,tag)]=0

        for sentence in tagged_sents:
            for tagged_word in sentence:
                self.counters[(tagged_word[0],tagged_word[1]) ]= self.counters[tagged_word] + 1
        for word in all_words:
            tags_counts={}
            for tag in (all_tags):
                tags_counts[tag]=self.counters[(word,tag)]
            self.words_highest_tag[word]=max(tags_counts, key=tags_counts.get)

        return

    def train(self, tagged_sents):
        self.count(tagged_sents)

    def tag_sentence(self, sent):
        tagged_sentece= np.zeros(np.shape(sent),dtype='O')
        for i in range(len(sent)):
            touple = (sent[i],self.words_highest_tag[sent[i]])
            tagged_sentece[i]=touple
        return tagged_sentece

class bigram(ngram):
    def count(self, tagged_sents):
        all_words, all_tags = get_all_possible_tags()

        single_word_dict = {}
        for tag in all_tags:
            single_word_dict[tag] = 0

        for word1 in all_words:
            for word2 in all_words:
                self.counters[(word1, word2)] = single_word_dict.copy()
        # todo add at any sentencs begining the * word, add * also in the counters
        # todo update counters
        return


model_uni = unigram()
model_uni.count(training_set)
#print(type(training_set))
for i in data:
    if i == None:
        print('none')
a = model_uni.test(data)
#sentence = np.array(['the', 'a'])
#tagged_sentence = model_uni.tag_sentence(sentence)
#print(tagged_sentence)
#print(model_uni.words_highest_tag)
#model_uni.test(training_set)
