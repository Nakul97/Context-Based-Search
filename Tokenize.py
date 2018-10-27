from sklearn.datasets import fetch_20newsgroups
import nltk
from nltk.corpus import stopwords
import gensim 
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from collections import namedtuple
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)


stop_words = set(stopwords.words('english'))

vectors = []

contentTokensProc = []
globalTokens = []

dataContentArray = []

for data in twenty_train.data:
    dataContent = ""
    dataArray = data.split("\n")
    for i in range(1,len(dataArray)):
        dataContent+= dataArray[i]
        dataContent+="\n"
    stemmer = nltk.PorterStemmer()
    contentTokens = nltk.word_tokenize(dataContent)
    content = ""

    for token in contentTokens:
        stemmedToken = stemmer.stem(token)
        if stemmedToken.lower() not in stop_words:
            content+=stemmedToken.lower()
            content+= " "
            
    dataContentArray.append(content)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(dataContentArray)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
predicted = text_clf.predict_proba(["Do you demand a TV with the most possible pixels even if there’s really nothing to watch to take advantage of all that resolution? Well, Samsung’s new 85-inch Q900 (which debuted back at CES in January) is ready to absorb the bonus you got for working at an investment bank. You can plunk down $15,000 and get yourself on the list for the fanciest TV around. The rest of us will have to endure our 4K displays for a few years before 8K becomes more mainstream."])
print(predicted)
joblib.dump(text_clf, 'MultinomialNB.joblib')
