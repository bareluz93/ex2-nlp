import nltk
from nltk.corpus import brown

data=brown.tagged_sents(categories='news')
training_set=data[0:round(len(data)*0.9)]
test_set=data[round(len(data)*0.9)+1:round(len(data)]