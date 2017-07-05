import mykNN
import matplotlib.pyplot as plt
from numpy import *

datingDataMat, datingLabels = mykNN.file2matrix('datingTestSet2.txt')
normDataSet, ranges, minVals = mykNN.autoNorm(datingDataMat)
figure = plt.figure()
ax = figure.add_subplot(111)
#Why?
ax.scatter(normDataSet[:, 1], normDataSet[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
plt.show()
mykNN.datingClassTest()