import numpy as np
import theano.tensor as T
import theano
import lasagne
import sys

from .. import Convolution2DSwitchLayer
from .. import DensePartialSwitchLayer
from .. import DenseSwitchLayer
from .. import DeepNetworks
from .. import DeepQTransferNetwork

def buildNeuralNetwork(batchSize, inputState, numOutputs, numTasks, convImplementation = "conv", layerNonlinearity = lasagne.nonlinearities.rectify):
    transferLayers = []
    networkInput = lasagne.layers.InputLayer(shape=((None,)+inputState))

    if convImplementation == "conv" or convImplementation == "dnn":
        if convImplementation == "conv":
            convFunction = lasagne.layers.conv.Conv2DLayer
        elif convImplementation == "dnn":
            from lasagne.layers import dnn
            convFunction = dnn.Conv2DDNNLayer

        conv1 = convFunction(
            networkInput, 
            num_filters = 32, 
            filter_size = (8,8), 
            stride=(4,4), 
            nonlinearity=layerNonlinearity, 
            W = lasagne.init.HeUniform(),
            b = lasagne.init.Constant(.1))

        conv2 = convFunction(
            conv1, 
            num_filters = 64, 
            filter_size = (4,4), 
            stride=(2,2), 
            nonlinearity=layerNonlinearity, 
            W = lasagne.init.HeUniform(),
            b = lasagne.init.Constant(.1))

        conv3 = convFunction(
            conv2, 
            num_filters = 64, 
            filter_size = (3,3), 
            stride=(1,1), 
            nonlinearity=layerNonlinearity, 
            W = lasagne.init.HeUniform(),
            b = lasagne.init.Constant(.1))

    elif convImplementation == "cuda":
        from lasagne.layers import cuda_convnet
        convFunction = cuda_convnet.Conv2DCCLayer
        dimshuffle = True
        c01b=True

        conv1 = convFunction(
            networkInput, 
            num_filters = 32, 
            filter_size = (8,8), 
            stride=(4,4), 
            nonlinearity=layerNonlinearity, 
            W = lasagne.init.HeUniform(c01b),
            b = lasagne.init.Constant(.1),
            dimshuffle=dimshuffle)

        conv2 = convFunction(
            conv1, 
            num_filters = 64, 
            filter_size = (4,4), 
            stride=(2,2), 
            nonlinearity=layerNonlinearity, 
            W = lasagne.init.HeUniform(c01b),
            b = lasagne.init.Constant(.1),
            dimshuffle=dimshuffle)

        conv3 = convFunction(
            conv2, 
            num_filters = 64, 
            filter_size = (3,3), 
            stride=(1,1), 
            nonlinearity=layerNonlinearity, 
            W = lasagne.init.HeUniform(c01b),
            b = lasagne.init.Constant(.1),
            dimshuffle=dimshuffle)


    hiddenLayer = lasagne.layers.DenseLayer(
        conv3,
        num_units=512,
        nonlinearity=layerNonlinearity,
        W = lasagne.init.HeUniform(),
        b = lasagne.init.Constant(.1))

    outputLayer = DenseSwitchLayer.DenseSwitchLayer(
        hiddenLayer,
        numSwitchOptions = numTasks,
        numSwitchedUnits = numOutputs,
        switchDefault = 0,
        W = lasagne.init.HeUniform(),
        b = lasagne.init.Constant(.1), 
        nonlinearity=None)

    transferLayers.append(outputLayer)

    return outputLayer, transferLayers