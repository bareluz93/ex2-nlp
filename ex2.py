import nltk
from nltk.corpus import brown
import numpy as np

data = brown.tagged_sents(categories='news')
training_set = data[0:round(len(data) * 0.9)]


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
        print('TBD')

    # implement in the derived
    def tag_sentence(self, sent):
        print('not implemented!')

    # get sentences, tag them using tag_sentence(), compare to original, compute and return the accuarcy
    def test(self, tagged_sents):
        print('TBD')

    # the full training method, implement in derived
    def train():
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
    words_highest_tag={}
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
    def tag_sentence(self, sent):
        tagged_sentece= np.zeros(np.shape(sent),dtype='str,str')
        for i in range(len(sent)):
            tagged_sentece[i]=(sent[i],self.words_highest_tag[sent[i]])
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
print(model_uni.words_highest_tag)
