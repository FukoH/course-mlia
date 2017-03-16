'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: pbharrin
'''

from numpy import *
import operator
from os import listdir

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn import decomposition, metrics
import time

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import csv

from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()     
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def file2matrix(filename):
    love_dictionary={'largeDoses':3, 'smallDoses':2, 'didntLike':1}
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)            #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return   
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        if(listFromLine[-1].isdigit()):
            classLabelVector.append(int(listFromLine[-1]))
        else:
            classLabelVector.append(love_dictionary.get(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

    
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals
   
def datingClassTest():
    hoRatio = 0.50      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
    print errorCount
    
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input(\
                                  "percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream, ])
    classifierResult = classify0((inArr - \
                                  minVals)/ranges, normMat, datingLabels, 3)
    print "You will probably like this person: %s" % resultList[classifierResult - 1]
    
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))


PCA_COMPONENTS = 100
DIGIT_RECOGNIZER_TRAIN_FILE = "/Users/liminghao/Downloads/train.csv"
DIGIT_RECOGNIZER_TEST_FILE = "/Users/liminghao/Downloads/test.csv"
def digitRecognizerTestFast():
    ## load train data
    train_data = pd.read_csv(DIGIT_RECOGNIZER_TRAIN_FILE)
    features_train = train_data.columns[1:]
    X_train = train_data[features_train]
    y_train = train_data['label']

    ## split train data into parts of train and test
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
    y_train = y_train.values
    y_test = y_test.values

    ## pca decomposition
    pca = decomposition.PCA(n_components=PCA_COMPONENTS).fit(X_train)
    X_train_reduced = pca.transform(X_train)

    values_dict = {}
    accuracy_dict = {}

    # for k in [3, 5, 10, 20]:
    for k in [3]:
        start_time = time.time()
        print "FOR K= ", k
        clf = KNeighborsClassifier(k)
        clf.fit(X_train_reduced, y_train)

        X_test_reduced = pca.transform(X_test)

        y_pred = clf.predict(X_test_reduced)

        values_dict[k] = y_pred

        print classification_report(y_test, y_pred)
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, y_pred))

        acc = accuracy_score(y_test, y_pred)
        accuracy_dict[k] = acc

        print("\n")
        print("Accuracy:%f" % (acc * 100))

        print("Runtime:")
        print round((time.time() - start_time), 2), " seconds"

    max_key = max(accuracy_dict, key=accuracy_dict.get)

    with open('KNN_results.csv', 'w')as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(["ImageID", "Label"])
        i = 1
        for v in values_dict[max_key]:
            writer.writerow([i, v])
            i += 1

def digitRecognizerForKaggle():
    ## load train data
    train_data = pd.read_csv(DIGIT_RECOGNIZER_TRAIN_FILE)
    features_train = train_data.columns[1:]
    X_train = train_data[features_train]
    y_train = train_data['label']
    y_train = y_train.values


    ## load test data
    test_data = pd.read_csv(DIGIT_RECOGNIZER_TEST_FILE)
    features_test = test_data.columns[0:]
    X_test = test_data[features_test]

    ## pca decomposition
    pca = decomposition.PCA(n_components=PCA_COMPONENTS).fit(X_train)
    X_train_reduced = pca.transform(X_train)

    # for k in [3, 5, 10, 20]:
    for k in [3]:
        start_time = time.time()
        print "FOR K= ", k
        clf = KNeighborsClassifier(k)
        clf.fit(X_train_reduced, y_train)

        # X_test_reduced = pca.transform(X_test)[:1000]
        X_test_reduced = pca.transform(X_test)
        print "Start testing %d samples" % X_test_reduced.shape[0]
        y_pred = clf.predict(X_test_reduced)

        with open('KNN_%d_kaggle_results.csv' % k , 'w')as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(["ImageID", "Label"])
            for i, v in enumerate(y_pred):
                writer.writerow([i+1, v])

        print("Runtime:")
        print round((time.time() - start_time), 2), " seconds"

def digitRecognizerTest():
    ## load train data
    train_data = pd.read_csv(DIGIT_RECOGNIZER_TRAIN_FILE)
    train_data = shuffle(train_data)[:10000] # select 10000 data to test digit recognizer
    features_train = train_data.columns[1:]
    X_train = train_data[features_train]
    y_train = train_data['label']

    ##load test data
    # test_data = pd.read_csv(DIGIT_RECOGNIZER_TEST_FILE)
    # features_test = test_data.columns[1:]
    # X_test = test_data[features_test]
    # y_test = test_data['label']

    ## split train data into parts of train and test
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
    y_train = y_train.values
    y_test = y_test.values

    ## pca decomposition
    pca = decomposition.PCA(n_components=PCA_COMPONENTS).fit(X_train)
    X_train_reduced = pca.transform(X_train)

    values_dict = {}
    accuracy_dict = {}

    # for k in [3, 5, 10, 20]:
    for k in [3]:
        start_time = time.time()
        print "FOR K= ", k
        # clf = KNeighborsClassifier(k)
        # clf.fit(X_train_reduced, y_train)

        X_test_reduced = pca.transform(X_test)

        y_pred = []
        for j in X_test_reduced:
            classifierResult = classify0(j, X_train_reduced, y_train, k)
            y_pred.append(classifierResult)
            print "processed: %d/%d" % (len(y_pred), len(y_test))
        # y_pred = clf.predict(X_test_reduced)

        values_dict[k] = y_pred

        print classification_report(y_test, y_pred)
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, y_pred))

        acc = accuracy_score(y_test, y_pred)
        accuracy_dict[k] = acc

        print("\n")
        print("Accuracy:%f" % (acc * 100))

        print("Runtime:")
        print round((time.time() - start_time), 2), " seconds"

    max_key = max(accuracy_dict, key=accuracy_dict.get)

    with open('KNN_results.csv', 'w')as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(["ImageID", "Label"])
        i = 1
        for v in values_dict[max_key]:
            writer.writerow([i, v])
            i += 1

if __name__ == '__main__':
    ## test handwriting
    # handwritingClassTest()

    ## test digit recognizer using knn in book
    # digitRecognizerTest()

    ## test digit recognizer fast using knn in sklearn
    # digitRecognizerTestFast()

    ## test digit recognizer for kaggle
    digitRecognizerForKaggle()