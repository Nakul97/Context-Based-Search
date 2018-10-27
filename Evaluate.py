from sklearn.externals import joblib
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
import nltk
import numpy as np

stop_words = set(stopwords.words('english'))

clf = joblib.load('MultinomialNB.joblib')

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

userReadList = [1,990,2300,566,345,778,2,667]
timePerArticle = [4.6, 4.4, 4.2, 4.7, 0.1, 0.5, 4.5, 4.4] 

userProfile = [0,0,0,0,0,0]

totalTime = 0

index = 0

for articleNumber in userReadList:
    article = twenty_train.data[articleNumber]
    dataArray = article.split("\n")
    twenty_train.data[articleNumber]
    dataContent = ""
    print(twenty_train.data[articleNumber])
    print(article.count(' '))
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
    
    predicted = clf.predict_proba([content])
    print(predicted)
    print(twenty_train.target[articleNumber])

    totalTime += timePerArticle[index] 

    userProfile[0] += (predicted[0][15] + predicted[0][19]) * timePerArticle[index] #relegion
    userProfile[1] += (predicted[0][1] + predicted[0][2] + predicted[0][3] + predicted[0][4] + predicted[0][5]) * timePerArticle[index] #technology
    userProfile[2] += (predicted[0][7] + predicted[0][8]) * timePerArticle[index] #automobile
    userProfile[3] += (predicted[0][9] + predicted[0][10]) * timePerArticle[index] #sports
    userProfile[4] += (predicted[0][11] + predicted[0][12] + predicted[0][13] + predicted[0][14]) * timePerArticle[index] # science
    userProfile[5] += (predicted[0][0] + predicted[0][6] + predicted[0][16] + predicted[0][17] + predicted[0][18]) * timePerArticle[index] #others

for i in range(len(userProfile)):
    userProfile[i]/=(totalTime)

userNp = np.array(userProfile)
np.save("userProfile", userNp)
print(userProfile)
