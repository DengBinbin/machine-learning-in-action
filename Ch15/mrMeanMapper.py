# coding=utf-8
'''
Created on Feb 21, 2011
Machine Learning in Action Chapter 18
Map Reduce Job for Hadoop Streaming 
mrMeanMapper.py
@author: Peter Harrington
'''
#-----------------------------------------------------
#   作者：Peter
#   TODO:
#   学习时间：2016-06-01
#   语言：Python 2.7.6
#   环境：linux（ubuntu14.04）
#-----------------------------------------------------
import sys
from numpy import mat, mean, power

def read_input(file):
    for line in file:
        yield line.rstrip()
        
input = read_input(sys.stdin)#creates a list of input lines

input = [float(line) for line in input] #overwrite with floats
numInputs = len(input)
input = mat(input)
sqInput = power(input,2)

#output size, mean, mean(square values)
print "%d\t%f\t%f" % (numInputs, mean(input), mean(sqInput)) #calc mean of columns
print >> sys.stderr, "report: still alive" 
