#!usr/bin/env python2.7
"""
Author: renjie
Project: web crawler
"""

from numpy import *
import matplotlib.pyplot as plot
# from math import exp

"""
function: load file and read data to the data set
"""
def loadDataSet():
    dataMat = []
    labelMat = []
    fp = open("../data/testSet.txt")
    for line in fp.readlines():
        lineArr = line.strip().split() # remove the white space of the string
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmod(x):
    return 1.0/(1+exp(-x))


def grandAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001   # step size of the grand ascent algorithm
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmod(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


def plotBestFit(weight):
    try:
        weights = weight.getA()
    except Exception, e:
        print Exception, ":", e
        weights = weight
        
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])

    fig = plot.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plot.xlabel('X1')
    plot.ylabel('X2')
    plot.show()


"""
    stochastic gradient ascent

    1.start from the weights all set to one
    2.for each piece of data in the dataset:
        calculate the gradient of one piece of data
        Update the weights vector by alpha * gradient
        return the weights vector
"""
def stocGradAscent(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    weights = ones(n)
    alpha = 0.01
    for i in range(m):
        h = sigmod(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


if __name__ == "__main__":
    dataMat, labelMat = loadDataSet()  # load data
    weights = grandAscent(dataMat, labelMat) # compute the weights
    print weights
    # plotBestFit(weights) # plot line
    weightsStoc = stocGradAscent(array(dataMat), labelMat)
    print weightsStoc
    plotBestFit(weightsStoc)
