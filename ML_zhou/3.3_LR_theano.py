# coding: utf-8
import numpy as np
import theano.tensor as T
import theano
import numpy.random as rng
from theano.tensor.nnet import sigmoid, binary_crossentropy

#读入csv数据,每一行的格式都是1,0.697,0.46,是\n
file = open('data/西瓜数据集3.0.csv'.decode('utf-8'))
data = [raw.strip('\n').split(',') for raw in file]
X = [[float(raw[-3]), float(raw[-2])] for raw in data[1:]]
Y = [1 if raw[-1]=='是' else 0 for raw in data[1:]]

feats = len(X[0])
lrate = 1
maxturn = 10000
x = T.dmatrix('x')
y = T.vector('y')
w = theano.shared(rng.normal(size=feats), name='w')
b = theano.shared(rng.randn(), name='b')

z = T.dot(x, w) + b
p = sigmoid(z)
cost = binary_crossentropy(p, y).mean()
gw, gb = theano.grad(cost, [w, b])
pred_res = p > 0.5
fit = theano.function(inputs=[x, y], outputs=[cost, gw, gb], updates=((w, w - lrate * gw), (b, b - lrate * gb)))
predict = theano.function(inputs=[x], outputs=[pred_res])

for i in range(maxturn):
    print fit(X, Y)
train_res = predict(X)[0]
print 'predict result:'
print train_res
print 'accuracy:'
print float(sum([Y[i] == train_res[i] for i in range(len(Y))])) / len(Y)