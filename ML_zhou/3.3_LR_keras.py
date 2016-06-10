# coding: utf-8

"""
    使用keras实现logistic分类器
"""
import os
import gzip
import urllib

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
import keras
from keras.utils.np_utils import to_categorical
from  keras.callbacks import ModelCheckpoint,Callback
current_dir = os.path.abspath(os.path.curdir)
np.random.seed(1337)  # 每一次运行结果都一样
def load_data():
    #读入csv数据,每一行的格式都是1,0.697,0.46,是\n
    file = open('data/西瓜数据集3.0.csv'.decode('utf-8'))
    data = [raw.strip('\n').split(',') for raw in file]
    #X是密度和含糖率两项数据，Y是西瓜是否好瓜的label,1是0否
    X = [[float(raw[1]), float(raw[2])] for raw in data[1:]]
    Y = [1 if raw[-1]=='是' else 0 for raw in data[1:]]
    return X,Y

def create_model(input_dim, output_dim):
    """
    创建logistic模型
    :param input_dim: (int) 输入维度
    :param output_dim: (int) 输出维度
    :return: Sequential
    """
    model = Sequential()
    model.add(Dense(output_dim, input_dim=input_dim))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.05)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=["accuracy"])
    return model

if __name__ == '__main__':

    train_set_x, train_set_y = load_data()

    train_set_x = np.asarray(train_set_x)
    train_set_y = np.asarray(train_set_y)
    input_dim = train_set_x.shape[1]
    output_dim = train_set_y.max() - train_set_y.min() + 1
    train_set_y = to_categorical(train_set_y)
    #validate_set_y = to_categorical(validate_set_y)
    #test_set_y = to_categorical(test_set_y)

    model = create_model(input_dim, output_dim)
    checkpointer = ModelCheckpoint(filepath="best_model.hdf5", monitor='val_acc', verbose=1, save_best_only=True)
    # history = LossHistory()
    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_acc', patience=30, verbose=1, mode='auto')
    history = model.fit(train_set_x,train_set_y, batch_size=3, nb_epoch=200, callbacks=[checkpointer, earlyStopping],
                        shuffle=True, verbose=0,
                        validation_data=(train_set_x,train_set_y))
    result = model.predict(train_set_x,batch_size=20,verbose=1)
    result = [np.argmax(i) for i in result]
    print result
    score = model.evaluate(train_set_x, train_set_y, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
