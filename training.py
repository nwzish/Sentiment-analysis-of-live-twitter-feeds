import csv
import collections
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.metrics.scores import precision , recall , f_measure
from nltk.metrics import ConfusionMatrix
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from sklearn.linear_model import LogisticRegression , SGDClassifier
from sklearn.svm import LinearSVC,NuSVC
from sklearn.naive_bayes import MultinomialNB , BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from statistics import mode
import pickle

numberOfTweets = 40000
numberOfFeatures = 3000
print("No of Tweets: ",numberOfTweets , "No of Features: ",numberOfFeatures)


def ReadData(fileName="dataset.csv"):
    with open(fileName) as csvfile:
        data = csv.reader(csvfile,delimiter=",")
        tweets=[(tweet[0],tweet[1]) for tweet in data]

    return tweets[1:]

def WordFeatures(document):
    all_words = []
    lemmatizer = WordNetLemmatizer()
    for tweet in document:
        # print(tweet)
        # all_words = [word.lower() if word.lower() not in customStopwords for word in word_tokenize(tweet[0])]
        for word in word_tokenize(tweet[0]):
            # print(word)
            if word.lower() not in customStopwords:
                # print(word.lower())
                all_words.append(lemmatizer.lemmatize(word.lower()))
                # print(all_words)
    # print(all_words)
    all_words = nltk.FreqDist(all_words)
    # print(all_words.most_common())
    word_features = list(all_words.keys())[:numberOfFeatures]
    return word_features

def FindFeatures(tweet):
    words = []
    lemmatizer = WordNetLemmatizer()
    for word in word_tokenize(tweet):
        # print(word)
        if word.lower() not in customStopwords:
            # print(word.lower())
            words.append(lemmatizer.lemmatize(word.lower()))
    # print(words)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

class VotedClassifier(ClassifierI):
    def __init__(self,*classifiers):
        self._classifiers = classifiers

    def Classify(self,features):
        vote = []
        for c in self._classifiers:
            v = c.classify(features)
            # print(v)
            vote.append(v)
        return mode(vote)
    def Confidence(self,features):
        vote = []
        for c in self._classifiers:
            v = c.classify(features)
            vote.append(v)
        res_vote = vote.count(mode(vote))
        confidence = res_vote/len(vote)
        return confidence

def classification_result(classifier , test_set):
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
    reflist = []
    testlist = []
    for i , (tweet , label) in enumerate(test_set):
        refsets[label].add(i)
        reflist.append(label)
        observed = classifier.classify(tweet)
        testsets[observed].add(i)
        testlist.append(observed)
    print(len(refsets['1']))
    print("Accuracy : " ,nltk.classify.accuracy(classifier,test_set)*100)
    print("Precision Pos: " , precision(refsets['1'],testsets['1'])*100)
    print("Recall Pos: ", recall(refsets['1'],testsets['1'])*100)
    print("F Measure Pos: " , f_measure(refsets['1'], testsets['1'])*100)
    print("Precision Neg: ", precision(refsets['0'], testsets['0']) * 100)
    print("Recall Neg: ", recall(refsets['0'], testsets['0']) * 100)
    print("F Measure Neg: ", f_measure(refsets['0'], testsets['0']) * 100)
    print("Confusion Metrics : \n" , ConfusionMatrix(reflist,testlist))

def voted_classification_result(classifier , test_set):
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
    reflist = []
    testlist = []
    for i , (tweet , label) in enumerate(test_set):
        refsets[label].add(i)
        reflist.append(label)
        observed = classifier.Classify(tweet)
        testsets[observed].add(i)
        testlist.append(observed)
    print(len(refsets['0']))
    # print("Accuracy : " ,nltk.classify.accuracy(classifier,test_set)*100)
    print("Precision Pos: ", precision(refsets['1'], testsets['1']) * 100)
    print("Recall Pos: ", recall(refsets['1'], testsets['1']) * 100)
    print("F Measure Pos: ", f_measure(refsets['1'], testsets['1']) * 100)
    print("Precision Neg: ", precision(refsets['0'], testsets['0']) * 100)
    print("Recall Neg: ", recall(refsets['0'], testsets['0']) * 100)
    print("F Measure Neg: ", f_measure(refsets['0'], testsets['0']) * 100)
    print("Confusion Metrics : \n", ConfusionMatrix(reflist, testlist))

# Reading Data
tweets = ReadData("datasetnew.csv")

# Pickling data
save_tweets = open("trained_classifier/tweets.pickle","wb")
pickle.dump(tweets,save_tweets)
save_tweets.close()

# print(tweets[:20])

# Making custom list of stopwords
customStopwords = set(stopwords.words('english')+list(punctuation))
customStopwords.add("@")

# Making list of most frequent words
word_features = WordFeatures(tweets[:int(numberOfTweets*3/4)])

# Saving MostFrequent Words
save_word_features = open("trained_classifier/word_features.pickle","wb")
pickle.dump(word_features,save_word_features)
save_word_features.close()

# print(word_features)

# Making feature array
featureSets = [(FindFeatures(tweet),sentiment) for (tweet,sentiment) in tweets[:numberOfTweets]]
# print(featureSets[:10])

trainingSet = featureSets[:int(numberOfTweets*3/4)]
testingSet = featureSets[int(numberOfTweets*3/4):]


classifier = nltk.NaiveBayesClassifier.train(trainingSet)
# print("NlTK NaiveBayes  : ",nltk.classify.accuracy(classifier,testingSet)*100)
# classifier.show_most_informative_features()
classification_result(classifier , testingSet)

save_clf = open("trained_classifier/originalnaivebayes.pickle","wb")
pickle.dump(classifier, save_clf)
save_clf.close()

classifierMNB = SklearnClassifier(MultinomialNB())
classifierMNB.train(trainingSet)
# print("MultinomialNB    : ",nltk.classify.accuracy(classifierMNB,testingSet)*100)
print("MultinomialNB")
classification_result(classifierMNB,testingSet)

save_clf = open("trained_classifier/MultinomialNB.pickle","wb")
pickle.dump(classifierMNB, save_clf)
save_clf.close()

classifierBNB = SklearnClassifier(BernoulliNB())
classifierBNB.train(trainingSet)
# print("BernaulliNB  : ", nltk.classify.accuracy(classifierBNB,testingSet))
print("BernaulliNB")
classification_result(classifierBNB,testingSet)

save_clf = open("trained_classifier/BernoulliNB.pickle","wb")
pickle.dump(classifierBNB, save_clf)
save_clf.close()

classifierLR = SklearnClassifier(LogisticRegression())
classifierLR.train(trainingSet)
# print("LogisticRegression   : " , nltk.classify.accuracy(classifierLR,testingSet))
print("LogisticRegression")
classification_result(classifierLR,testingSet)

save_clf = open("trained_classifier/LogisticRegression.pickle","wb")
pickle.dump(classifierLR, save_clf)
save_clf.close()

classifierSGDC = SklearnClassifier(SGDClassifier())
classifierSGDC.train(trainingSet)
# print("SGDClassifier   : ", nltk.classify.accuracy(classifierSGDC,testingSet))
print("SGDClassifier")
classification_result(classifierSGDC,testingSet)

save_clf = open("trained_classifier/SGDClassifier.pickle","wb")
pickle.dump(SGDClassifier, save_clf)
save_clf.close()

classifierLSVC = SklearnClassifier(LinearSVC())
classifierLSVC.train(trainingSet)
# print("LinearSVC    : ", nltk.classify.accuracy(classifierLSVC,testingSet))
print("LinearSVC")
classification_result(classifierLSVC,testingSet)

save_clf = open("trained_classifier/LinearSVC.pickle","wb")
pickle.dump(classifierLSVC,save_clf)
save_clf.close()

classifierNSVC = SklearnClassifier(NuSVC())
classifierNSVC.train(trainingSet)
# print("NuSVC    : ",nltk.classify.accuracy(classifierNSVC,testingSet))
print("NuSVC")
classification_result(classifierNSVC,testingSet)

save_clf = open("trained_classifier/NuSVC.pickle","wb")
pickle.dump(classifierNSVC, save_clf)
save_clf.close()

classifierKNC = SklearnClassifier(KNeighborsClassifier(n_neighbors=10))
classifierKNC.train(trainingSet)
# print("KNeighboursClassifier    : ",nltk.classify.accuracy(classifierKNC,testingSet))
print("KNeighboursClssifier")
classification_result(classifierKNC,testingSet)

save_clf = open("trained_classifier/KNeighboursClassifier.pickle","wb")
pickle.dump(classifierKNC, save_clf)
save_clf.close()

classifierRFC = SklearnClassifier(RandomForestClassifier())
classifierRFC.train(trainingSet)
# print("RandomForestClassifier    : ",nltk.classify.accuracy(classifierRFC,testingSet))
print("RandomForestClassifier")
classification_result(classifierRFC,testingSet)

save_clf = open("trained_classifier/RandomForestClassifier.pickle","wb")
pickle.dump(classifierRFC, save_clf)
save_clf.close()

voted_classifier = VotedClassifier(classifier,
                                   classifierMNB,
                                   classifierBNB,
                                   classifierLR,
                                   classifierSGDC,
                                   classifierLSVC,
                                   classifierNSVC,
                                   classifierKNC,
                                   classifierRFC
                                   )
print("VotedClassifier")
voted_classification_result(voted_classifier,testingSet)

def FindSentiment(tweet):
    print("CLASSIFIER CALLED")
    feature = FindFeatures(tweet)
    # print(feature)
    sentiment = voted_classifier.Classify(feature)
    confidence = voted_classifier.Confidence(feature)
    print(sentiment,confidence)
    return sentiment,confidence

for i in range(10):
    print("classification   : ", voted_classifier.Classify(testingSet[i][0])," confidence   : ",voted_classifier.Confidence(testingSet[i][0]))
# print(word_features)

