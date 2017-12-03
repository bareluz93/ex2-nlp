__author__ = 'ssbushi'

# Import the toolkit and tags
import nltk
from nltk.corpus import brown
# Train data - pretagged
tmp = brown.sents(categories=['news'])
data = brown.tagged_sents(categories='news')
training_set = data[0:int(len(data) * 0.9)]
# test_set = tmp[int(len(data) * 0.9): len(data)]
test_set = data[int(len(data) * 0.9): len(data)]


train_data = training_set

print(train_data[0])

# Import HMM module
from nltk.tag import hmm

# Setup a trainer with default(None) values
# And train with the data
trainer = hmm.HiddenMarkovModelTrainer()
tagger = trainer.train_supervised(train_data)

tagger.test(train_data)
# print(tagger.evaluate(test_set))