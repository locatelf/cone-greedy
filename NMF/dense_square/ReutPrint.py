import numpy as np  
import matplotlib.pyplot as plt    
import nltk
from nltk.corpus import reuters  
#print(len(reuters.words()))  
vocabulary = set(reuters.words())  
stopwords = nltk.corpus.stopwords.words()  
cleansed_words = [w.lower() for w in reuters.words() if w.isalnum() and w.lower() not in stopwords]  
vocabulary = set(cleansed_words)  
files = [f for f in reuters.fileids() if 'training' in f]  
corpus = [reuters.raw(fileids=[f]) for f in files]  
from sklearn.feature_extraction.text import CountVectorizer  
vectorizer = CountVectorizer(stop_words='english')  
X = vectorizer.fit_transform(corpus)  
X.toarray()
print(X.toarray()[1])
np.savetxt("reuters.csv", X.toarray(), delimiter=",")

