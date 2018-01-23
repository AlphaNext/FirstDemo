#!/usr/bin/env python
# coding: utf-8

import time
import numpy as np
import mnist
from mnist import read

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def initParam():
    params = list()
    rng = np.random.RandomState()
    W0 = rng.normal(size=(28*28, 512), scale=0.1)
    W1 = rng.normal(size=(512, 64), scale=0.1)
    W2 = rng.normal(size=(64, 10), scale=0.1)
    B0 = np.zeros(512)
    B1 = np.zeros(64)
    B2 = np.zeros(10)

    params.append(W0)
    params.append(B0)
    params.append(W1)
    params.append(B1)
    params.append(W2)
    params.append(B2)
    return params

def fprop(input, param):
    shape_in = input.shape
    shape_after = shape_in[1] * \
                  shape_in[2] * \
                  shape_in[3]
    temp_In = input.reshape(shape_in[0],shape_after)
    midlayers = list()
    output1 = sigmoid(np.dot(temp_In, param[0]) + param[1])
    output2 = sigmoid(np.dot(output1, param[2]) + param[3])
    output3 = sigmoid(np.dot(output2, param[4]) + param[5])

    # midlayers.append(output3)
    midlayers.append(output2)
    midlayers.append(output1)
    midlayers.append(temp_In)

    pred = output3
    return pred, midlayers, param


def bprop(grad, midlayers, params):
    dW2 = np.dot(midlayers[0].T, grad)
    db2 = np.mean(grad, axis=0)
    grad2 = np.dot(grad, params[-2].T) * midlayers[0] * (1-midlayers[0])
    dW1 = np.dot(midlayers[1].T, grad2)
    db1 = np.mean(grad2, axis=0)
    grad3 = np.dot(grad2, params[2].T) * midlayers[1] * (1-midlayers[1])
    dW0 = np.dot(midlayers[2].T, grad3)
    db0 = np.mean(grad3, axis=0)
    backgrad = list()
    backgrad.append(dW0)
    backgrad.append(db0)
    backgrad.append(dW1)
    backgrad.append(db1)
    backgrad.append(dW2)
    backgrad.append(db2)
    return backgrad

def update_param(param, backgrad, learningRate):
    updated = list()
    for item, ditem in zip(param, backgrad):
        item -= learningRate*ditem
        updated.append(item)
    return updated

def prepare_data():
    # prepare data
    training_data = list(read(dataset='training', path='./'))
    testing_data = list(read(dataset='testing', path='./'))
    split = len(training_data)
    label, pixels = training_data[0]
    train_data = np.zeros(shape= (len(training_data), 1) + pixels.shape)
    train_label = np.zeros(shape = (len(training_data), 10))
    for n in range(len(training_data)):
        train_label[n, training_data[n][0]] = 1
        train_data[n, 0, :, :] = training_data[n][1] / 255.0

    Te_label, Te_pixels = testing_data[0]
    test_data = np.zeros(shape= (len(testing_data), 1) + Te_pixels.shape)
    test_label = np.zeros(shape = (len(testing_data), 10))
    for n in range(len(testing_data)):
        test_label[n, testing_data[n][0]] = 1
        test_data[n, 0, :, :] = testing_data[n][1] / 255.0

    # Downsample training data
    n_train_samples = 30000
    train_idxs = np.random.random_integers(0, split-1, n_train_samples)
    train_data = train_data[train_idxs, ...]
    train_label = train_label[train_idxs, ...]

    return train_data, train_label, test_data, test_label

def crossEntropyCost(input_data, label):
    C = label * np.log(input_data)
    C = np.sum(C, axis=1)
    C = np.mean(C, axis=0)
    return -C

def error(X, labels, params):
    changed_label = np.zeros(labels.shape[0])
    for i in range(labels.shape[0]):
        temp = list(labels[i, :])
        changed_label[i] = int(temp.index(1))
    Y_pred, midlayers, param= fprop(X, params)
    index = np.argmax(Y_pred, axis = 1)
    error = index != changed_label
    return np.mean(error)

def run():
    train_data, train_label, test_data, test_label = prepare_data()
    learning_rate = 0.01
    max_iter = 30
    batch_size = 128
    n_samples = train_data.shape[0]
    split = 5000
    val_data = train_data[n_samples - split:n_samples, ...]
    val_label = train_label[n_samples - split:n_samples, ...]
    train_data = train_data[0:n_samples - split, ...]
    train_label = train_label[0:n_samples - split, ...]
    n_batches = train_data.shape[0] // batch_size
    iter = 0
    updated_params = initParam()
    while iter < max_iter:
        iter += 1
        for b in range(n_batches):
            batch_begin = b * batch_size
            batch_end = batch_begin + batch_size
            data_batch = train_data[batch_begin:batch_end]
            label_batch = train_label[batch_begin:batch_end]
            pred, midlayers, params = fprop(data_batch, updated_params)
            grad = pred - label_batch
            backgrad = bprop(grad, midlayers, params)
            updated_params = update_param(params, backgrad, learning_rate)
            cross_loss = crossEntropyCost(pred, label_batch)
            print ('Epoch %d: Batch percentage %d / %d,  crossEntropy loss %.4f ' % (iter, b + 1, n_batches, cross_loss))
        val_err = error(val_data, val_label, updated_params)
        accuracy = 1 - val_err
        print ('Epoch %i: val accuracy %.4f' % (iter, accuracy))
    # test the trained model
    test_err = error(test_data, test_label, updated_params)
    print ('Test model accuracy %.4f' % (1 - test_err))

if __name__ == '__main__':
    run()
