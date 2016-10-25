import nltk
import random
from nltk.corpus import stopwords
doc=open('final.txt')
d=doc.read()
corpus=d.split()
#print(len(corpus))

stopset = list(set(stopwords.words('english')))
sem=';'
sem1=sem.split()
stop=stopset+sem1
def word_feats(words):
    return dict([(word, True) for word in words.split() if word not in stop])
##    return dict([(word, True) for word in words.split() if word not in stopset])
#pos_feats = [(word_feats(f)) for f in corpus]
#neg_feats = [(word_feats(f)) for f in corpus]

i=0
l1=[]
feats=[]
l2=[]
for item in corpus:
	if i%2==0:
		l1.append(str(item))
		feats.append(word_feats(item))
	else:
		l2.append(str(item))
	i=i+1
	
corpus1=zip(feats,l2)
#print(len(corpus1))


#print(corpus2)
training_set=corpus1[0:4329]				#60% of data
dev_set=corpus1[4329:5772]				#20% of data
test_set=corpus1[5772:7215]				#20% OF data

f=raw_input("Enter sentence to be classified:")
#print(test_set)
classifier=nltk.NaiveBayesClassifier.train(training_set)
print(classifier.classify(word_feats(f)))
#print(nltk.classify.accuracy(classifier, dev_set))
#print(classifier.show_most_informative_features(5))
