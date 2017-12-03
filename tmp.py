from nltk.model import NgramModel
from nltk.corpus import brown
from nltk.probability import LaplaceProbDist

est = lambda fdist: LaplaceProbDist(fdist)

corpus = brown.words(categories='news')[:100]
lm = NgramModel(3, corpus, estimator=est)


print(lm)
print(corpus[8], corpus[9], corpus[12] )
print(lm.prob(corpus[12], [corpus[8], corpus[9]]) )
print