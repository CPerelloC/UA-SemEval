import re
import spacy
from spacy.tokenizer import Tokenizer
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from textblob import TextBlob
import gensim
from gensim.models import KeyedVectors
import csv
import numpy as np
from collections import Counter
import string
import math
import operator
from math import log,exp,factorial,log10,pi
from sklearn.feature_selection import VarianceThreshold

def stirling(n):
    return n*log(n)-n +(0.5*log(2*3.141592*n))

def lexical_specificity(T,t,f,k):
    lim=400
    expvalue=f*t/(T*1.0)
    small=False
    if (expvalue<k):
        if (f>lim):
            arg1=stirling(f)
        else:
            arg1=log(factorial(int(f)))
        if (T-f>lim):
            arg2=stirling(T-f)
        else:
            arg2=log(factorial(int(T-f)))
        if (t>lim):
            arg3=stirling(t)
        else:
            arg3=log(factorial(int(t)))            
        if (T-t>lim):
            arg4=stirling(T-t)
        else:
            arg4=log(factorial(int(T-t)))
        if (T>lim):
            arg5=stirling(T)
        else:
            arg5=log(factorial(int(T)))
        if (k>lim):
            arg6=stirling(k)
        else:
            arg6=log(factorial(int(k)))
        if (f-k>lim):
            arg7=stirling(f-k)
        else:
            arg7=log(factorial(int(f-k)))
        if (t-k>lim):
            arg8=stirling(t-k)
        else:
            arg8=log(factorial(int(t-k)))
        if (T-f-t+k>lim):
            arg9=stirling(T-f-t+k)
        else:
            arg9=log(factorial(int(T-f-t+k)))

        prob=arg1+arg2+arg3+arg4-arg5-arg6-arg7-arg8-arg9
        first_prod=-log10(math.e)*prob
        
        if(prob<log(0.1)):
            small=True
            prob1=1.0
            prob=1.0
            while (prob1!=0.0 and (prob/prob1)<10000000 and k<=f):                
                prob2=(prob1*(f-k)*(t-k))/((k+1)*(T-f-t+k+1))
                prob=prob+prob2
                prob1=prob2
                k+=1
    if small:
        score=first_prod-log10(prob)
        return score
    else:
        return 0

## Load vectors | word embeddings -> return model
def load_wordembeddings_vectors(filename):
    return KeyedVectors.load_word2vec_format(filename)
    #'./Word_Embedding_Twitter/fasttext_spanish_twitter_100d.vec'

## Tokenizer utils
prefix_re = re.compile(r'''^[\[\(-.;?!/*+"']''')
suffix_re = re.compile(r'''[\]\)"-.;?!/*+']$''')
infix_re = re.compile(r'''[-.;?!/*+~]''')
simple_url_re = re.compile(r'''^https?://''')
def custom_tokenizer(nlp):
    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                token_match=simple_url_re.match)

## Read data -> vector, dict, resultsHS, resultsTR, resultsAG ids
def readData(filename):
    hate = "amariconada amariconadas amariconado amariconados amariconar aracuano bachiche bachiches bolillo bollera bolleras catalufos chola conguito conguitos culandron culandrones furcia furcias gitanada gabacho gitanadas gringa gringo india indio lagartona mongolo moro moromierda mujerzuela mujerzuelas negro perra pinche puta putona putaza sudaca sudacas zorra zorrita afrucan africano bitch black cunt cock sex kitten girly slut trixie"
    vector = []
    dictionary = {}
    resultsHS = []
    resultsTR = []
    resultsAG = []
    plurals = []
    hateWords = []
    ids = []
    f = open(filename)
    line = f.readline()
    while line != "":
        line = f.readline()
        cont = 0
        ht = 0
        if line != "":
            myId, textline, hs, tr, ag = line.rstrip().split('\t')
            resultsHS.append(hs)
            resultsTR.append(tr)
            resultsAG.append(ag)
            ids.append(myId)
            line = textline
            vector.append(line)
            nlpline = nlp(line)
            for token in nlpline:
                if (token.tag_).find('Plur') != -1:
                    cont = cont + 1
                if hate.find(token.text):
                    ht = ht + 1
                if token.text in dictionary:
                    value = dictionary.get(token.text)
                    value += 1
                    dictionary[token.text] = value
                else:
                    dictionary[token.text] = 0
            #Bi-gramas
            for i in range(0, len(nlpline)-1):
                if (nlpline[i].text, nlpline[i+1].text) in dictionary:
                    value = dictionary.get((nlpline[i].text, nlpline[i+1].text))
                    value += 2
                    dictionary[(nlpline[i].text, nlpline[i+1].text)] = value
                else:
                    dictionary[(nlpline[i].text, nlpline[i+1].text)] = 0
            #Tri-gramas
            for i in range(0, len(nlpline)-2):
                if (nlpline[i].text, nlpline[i+1].text, nlpline[i+2].text) in dictionary:
                    value = dictionary.get((nlpline[i].text, nlpline[i+1].text, nlpline[i+2].text))
                    value += 3
                    dictionary[(nlpline[i].text, nlpline[i+1].text, nlpline[i+2].text)] = value
                else:
                    dictionary[(nlpline[i].text, nlpline[i+1].text, nlpline[i+2].text)] = 0
            plurals.append(cont)
            hateWords.append(ht)
    f.close()
    return vector, dictionary, resultsHS, resultsTR, resultsAG, ids, plurals, hateWords


def readDataEv(filename):
    hate = "amariconada amariconadas amariconado amariconados amariconar aracuano bachiche bachiches bolillo bollera bolleras catalufos chola conguito conguitos culandron culandrones furcia furcias gitanada gabacho gitanadas gringa gringo india indio lagartona mongolo moro moromierda mujerzuela mujerzuelas negro perra pinche puta putona putaza sudaca sudacas zorra zorrita afrucan africano bitch black cunt cock sex kitten girly slut trixie"
    vector = []
    dictionary = {}
    plurals = []
    hateWords = []
    ids = []
    f = open(filename)
    line = f.readline()
    while line != "":
        line = f.readline()
        cont = 0
        ht = 0
        if line != "":
            myId, textline = line.rstrip().split('\t')
            ids.append(myId)
            line = textline
            vector.append(line)
            nlpline = nlp(line)
            for token in nlpline:
                if (token.tag_).find('Plur') != -1:
                    cont = cont + 1
                if hate.find(token.text):
                    ht = ht + 1
                if token.text in dictionary:
                    value = dictionary.get(token.text)
                    value += 1
                    dictionary[token.text] = value
                else:
                    dictionary[token.text] = 0
            #Bi-gramas
            for i in range(0, len(nlpline)-1):
                if (nlpline[i].text, nlpline[i+1].text) in dictionary:
                    value = dictionary.get((nlpline[i].text, nlpline[i+1].text))
                    value += 2
                    dictionary[(nlpline[i].text, nlpline[i+1].text)] = value
                else:
                    dictionary[(nlpline[i].text, nlpline[i+1].text)] = 0
            #Tri-gramas
            for i in range(0, len(nlpline)-2):
                if (nlpline[i].text, nlpline[i+1].text, nlpline[i+2].text) in dictionary:
                    value = dictionary.get((nlpline[i].text, nlpline[i+1].text, nlpline[i+2].text))
                    value += 3
                    dictionary[(nlpline[i].text, nlpline[i+1].text, nlpline[i+2].text)] = value
                else:
                    dictionary[(nlpline[i].text, nlpline[i+1].text, nlpline[i+2].text)] = 0
            plurals.append(cont)
            hateWords.append(ht)
    f.close()
    return vector, dictionary, ids, plurals, hateWords

## Erase the keys of the dictionary that doesn't appear N times -> Dictionary
def delimitDictionary(n, dictionary):
    dictAux = dictionary.copy()
    for key in dictAux:
        if dictAux[key] < n:
            dictionary.pop(key)
    return dictionary

## Establish an index in a dictionary -> Dictionary
def addIndex(dictionary):
    sum = 0
    for key in dictionary:
        dictTrain[key] = sum
        sum += 1
    return dictionary

def prepareVectors(vector, dictionary):
    X = []
    XL = []
    for line in vector:
        linetokenize = nlp(line)
        XL = [0 for i in range(len(dictionary))]
        #Unigrams and bigrams
        for i in range(0, len(linetokenize)-1):
            if linetokenize[i].text in dictionary:
                XL[dictionary.get(linetokenize[i].text)] += 1
            if (linetokenize[i].text, linetokenize[i+1].text) in dictionary:
                XL[dictionary.get((linetokenize[i].text, linetokenize[i+1].text))] += 1
        #Trigrams
        for i in range(0, len(linetokenize)-2):
            if (linetokenize[i].text, linetokenize[i+1].text, linetokenize[i+2].text) in dictionary:
                XL[dictionary.get((linetokenize[i].text, linetokenize[i+1].text, linetokenize[i+2].text))] += 1
        #Last unigram
        if linetokenize[len(linetokenize)-1].text in dictionary:
            XL[dictionary.get(linetokenize[len(linetokenize)-1].text)] += 1   
        X.append(XL.copy())
    return X

def calcAccuracy(vectorPredicted, results):
    it = 0
    for n in range(0, len(vectorPredicted)):
        if vectorPredicted[n] == results[n]:
            it += 1
    return (it / len(vectorPredicted))

def printResults(resultsHS, resultsTR, resultsAG, ids, filename):
    file = open(filename, "w")
    for n in range(0, len(resultsHS)):
        s = ids[n] + "\t" + str(resultsHS[n]) + "\t" + str(resultsTR[n]) + "\t" + str(resultsAG[n])
        file.write(s + '\n')
    file.close()

def sentimentAnalysis(vectorTrain, vector):
    for i in range(0, len(vectorTrain)):
        analysis = TextBlob(vectorTrain[i])
        vector[i].append(analysis.sentiment.polarity)
        vector[i].append(analysis.sentiment.subjectivity)
    return vector

def addTweetLength(vectorTrain, vector):
    for i in range(0, len(vectorTrain)):
        vector[i].append(len(vectorTrain[i]))
    return vector

def addPlurals(plurals, vector):
    for i in range(0, len(vector)):
        vector[i].append(plurals[i])
    return vector

def addHate(hate, vector):
    for i in range(0, len(vector)):
        vector[i].append(hate[i])
    return vector

def removingFeatures(X, Xdev):
    Y = X.copy()
    for i in range(0, len(XDev)):
        Y.append(XDev[i])
    delimitedY = VarianceThreshold(threshold=(.999 * (1 - .999))).fit_transform(Y)
    newX = [] 
    newXDev = []
    for i in range(0, len(X)):
        newX.append(delimitedY[i])
    for i in range(len(X), len(X) + len(XDev)):
        newXDev.append(delimitedY[i])
    return newX, newXDev

def wordEmbeddingsAverage(dictTrain, vector, model):
    vector_aumented = vector.copy()
    aux = list(model.wv.vocab)
    dictList = []
    for key in dictTrain:
        dictList.append(key)
    for i in range(0, len(vector)):
        for j in range(0, len(vector[i])):
            if vector[i][j] >= 1:
                if dictList[j] in aux:
                    a = np.array(model.get_vector(dictList[j]))
                    vector_aumented[i].append(np.average(a))
                else:
                    vector_aumented[i].append(0)
            else:
                vector_aumented[i].append(0)
    return vector_aumented

def SVMTrain(X, XDev, resultsTrain, resultsDev):
    Yresults = []
    acc = 0
    cont = 1
    for i in range(0, 6):
        print("C: " + str(cont))
        clf = svm.SVC(C=cont, kernel='linear', gamma='auto')
        clf.fit(X, resultsTrain)
        print("Classifier fit")
        #sorted(clf.cv_results_.keys())
        YPredict = clf.predict(XDev)
        print("Prediction finished")

        accuracy = calcAccuracy(YPredict, resultsDev)
        if accuracy > acc:
            acc = accuracy
            Yresults = YPredict.copy()
        print("Accuracy: " + str(accuracy))

        print("C: 2^-" + str(cont))
        clf = svm.SVC(C=2**-cont, kernel='linear', gamma='auto')
        clf.fit(X, resultsHSTrain)
        print("Classifier fit")
        #sorted(clf.cv_results_.keys())

        YPredict = clf.predict(XDev)
        print("Prediction finished")
        cont = cont + 1
        accuracy = calcAccuracy(YPredict, resultsDev)
        if accuracy > acc:
            acc = accuracy
            Yresults = YPredict.copy()
        print("Accuracy: " + str(accuracy))
    return Yresults
        #printResults(YPredict, ids, "es_a.tsv")
        #print("Results printed")
        #clf = GridSearchCV(svc, parameters, cv=5)

def SVMTrainEv(X, XDev, resultsTrain, ce):
    print("C: " + str(ce))
    clf = svm.SVC(C=ce, kernel='linear', gamma='auto')
    clf.fit(X, resultsTrain)
    print("Classifier fit")
    YPredict = clf.predict(XDev)
    return YPredict

nlp = spacy.load('en')
nlp.tokenizer = custom_tokenizer(nlp)
print("Tokenizer loaded")

#model = load_wordembeddings_vectors('./Word_Embedding_Twitter/fasttext_spanish_twitter_100d.vec')
#print("Word embeddings loaded")

vectorTrain, dictTrain, resultsHSTrain, resultsTRTrain, resultsAGTrain, idsTrain, pluralsTrain, hateTrain = readData("Data/SemEvalCodaLab/public_development_en/traindev_en.tsv")
print("Corpus train readed")
vectorDev, dictDev, ids, pluralsDev, hateDev = readDataEv("Data/SemEvalCodaLab/Evaluation/English/public_test_en/test_en.tsv")
print("Corpus test readed")

"""
frequency = {}
document_text = open('esp_tokenized.txt_noskin_final', 'r')
text_string = document_text.read().lower()
match_pattern = text_string.rstrip().split(' ')
 
for word in match_pattern:
    count = frequency.get(word,0)
    frequency[word] = count + 1
"""



# Data Info
print("Train HS: " + str(Counter(resultsHSTrain)))
print("Train TR: " + str(Counter(resultsTRTrain)))
print("Train AG: " + str(Counter(resultsAGTrain)))

#print("Dev HS: " + str(Counter(resultsHSDev)))
#print("Dev TR: " + str(Counter(resultsTRDev)))
#print("Dev AG: " + str(Counter(resultsAGDev)))

dictTrain = delimitDictionary(3, dictTrain)
print("Dictionary delimited")
dictTrain = addIndex(dictTrain)
print("Index established")
X = prepareVectors(vectorTrain, dictTrain)
print("Train vectors prepared")
XDev = prepareVectors(vectorDev, dictTrain)
print("Dev vectors prepared")


"""X = wordEmbeddingsAverage(dictTrain, X, model)
print("Train WordEmbeddings average added")
XDev = wordEmbeddingsAverage(dictTrain, XDev, model)
print("Dev WordEmbeddings average added")"""
"""
X = sentimentAnalysis(vectorTrain, X)
print("Train vectors sentiment analysis")
XDev = sentimentAnalysis(vectorDev, XDev)
print("Dev vectors sentiment analysis")
"""
#X = addTweetLength(vectorTrain, X)
#XDev = addTweetLength(vectorDev, XDev)
#print("Tweet length added")
print("Printo")
print(len(X))
print(len(pluralsTrain))

X = addPlurals(pluralsTrain, X)
XDev = addPlurals(pluralsDev, XDev)

X = addHate(hateTrain, X)
XDev = addHate(hateDev, XDev)

print("Length X: " + str(len(X[0])))
X, XDev = removingFeatures(X, XDev)
print("Length X: " + str(len(X[0])))



HS = SVMTrainEv(X, XDev, resultsHSTrain, 2**-5)
TR = SVMTrainEv(X, XDev, resultsTRTrain, 3)
AG = SVMTrainEv(X, XDev, resultsAGTrain, 2**-5)


for i in range(0, len(HS)):
    if HS[i] == '0':
        if TR[i] == '1' or AG[i] == '1':
            TR[i] = '0'
            AG[i] = '0'

"""accuracy = calcAccuracy(HS, resultsHSDev)
print("Accuracy HS: " + str(accuracy))
accuracy = calcAccuracy(TR, resultsTRDev)
print("Accuracy TR: " + str(accuracy))
accuracy = calcAccuracy(AG, resultsAGDev)
print("Accuracy AG: " + str(accuracy))"""


"""
def calcAccuracyB(HS, TR, AG, resultsHS, resultsTR, resultsAG):
    it = 0
    for n in range(0, len(HS)):
        if HS[n] == resultsHS[n] and TR[n] == resultsTR[n] and AG[n] == resultsAG[n]:
            it += 1
    return (it / len(HS))"""

#accuracy = calcAccuracyB(HS, TR, AG, resultsHSDev, resultsTRDev, resultsAGDev)
#print("Total accuracy: " + str(accuracy))

printResults(HS, TR, AG, ids, "en_a.tsv")








#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}


# Best result: C = 2^-4, kernel='linear', delimiter: 16



