import csv
import nltk
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from sklearn.linear_model import LogisticRegression , SGDClassifier
from sklearn.svm import LinearSVC,NuSVC
from sklearn.naive_bayes import MultinomialNB , BernoulliNB
from statistics import mode
import pickle
# import training

save_word_features = open("trained_classifier/word_features.pickle","rb")
word_features = pickle.load(save_word_features)
save_word_features.close()

class VotedClassifier(ClassifierI):
    def __init__(self,*classifiers):
        self._classifiers = classifiers

    def Classify(self,features):
        vote = []
        for c in self._classifiers:
            v = c.classify(features)
            print(v)
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


def FindFeatures(tweet):
    word = word_tokenize(tweet)
    feature = {}
    for w in word_features:
        feature[w] = (w in word)

    return feature

save_tweets = open("trained_classifier/tweets.pickle","rb")
tweets = pickle.load(save_tweets)
save_tweets.close()

load_classifier = open("trained_classifier/originalnaivebayes.pickle","rb")
classifier = pickle.load(load_classifier)
load_classifier.close()

load_classifier = open("trained_classifier/MultinomialNB.pickle","rb")
classifierMNB = pickle.load(load_classifier)
load_classifier.close()

load_classifier = open("trained_classifier/BernoulliNB.pickle","rb")
classifierBNB = pickle.load(load_classifier)
load_classifier.close()

load_classifier = open("trained_classifier/LogisticRegression.pickle","rb")
classifierLR = pickle.load(load_classifier)
load_classifier.close()

load_classifier = open("trained_classifier/SGDClassifier.pickle","rb")
classifierSGDC = pickle.load(load_classifier)
load_classifier.close()

load_classifier = open("trained_classifier/LinearSVC.pickle","rb")
classifierLSVC = pickle.load(load_classifier)
load_classifier.close()

load_classifier = open("trained_classifier/NuSVC.pickle","rb")
classifierNSVC = pickle.load(load_classifier)
load_classifier.close()

load_classifier = open("trained_classifier/KNeighboursClassifier.pickle","rb")
classifierKNC = pickle.load(load_classifier)
load_classifier.close()

load_classifier = open("trained_classifier/RandomForestClassifier.pickle","rb")
classifierRFC = pickle.load(load_classifier)
load_classifier.close()

voted_classifier = VotedClassifier(
                                    classifier,
                                   classifierMNB,
                                   classifierBNB,
                                   classifierLR,
                                   # classifierSGDC,
                                   classifierLSVC,
                                   classifierKNC,
                                   classifierRFC,
                                   classifierNSVC
                                  )

def FindSentiment(tweet):
    print("CLASSIFIER CALLED")
    feature = FindFeatures(tweet)
    # print(feature)
    sentiment = voted_classifier.Classify(feature)
    confidence = voted_classifier.Confidence(feature)
    print(sentiment,confidence)
    return sentiment,confidence

# for (tweet,sent) in tweets:
#     feature = FindFeatures(tweet)
#     FindSentiment(feature)



