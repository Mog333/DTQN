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
    
    #Merge type can be sum max or concat. For this network concat cant work as its an output layer 
    #This network defaults to sum for now and can be a parameter later if we want to play with it
    mergeType = 'sum'

    networkInput = lasagne.layers.InputLayer(shape=((None,)+inputState))

    if convImplementation != "conv" and convImplementation != "dnn":
        raise ValueError("Conv2DSwitch layer has only implemented theanos conv and cuDNN dnn convolutions. No support for CUDA (yet?)")

    
    if convImplementation == "conv":
        convFunction = lasagne.layers.conv.Conv2DLayer
    elif convImplementation == "dnn":
        from lasagne.layers import dnn
        convFunction = dnn.Conv2DDNNLayer



    conv1 = Convolution2DSwitchLayer.Conv2DSwitchLayer(
        networkInput,
        numSwitchOptions = numTasks,
        num_filters = 32,
        filter_size = (8,8),
        stride = (4,4),
        W = lasagne.init.HeUniform(),
        b = lasagne.init.Constant(.1),
        nonlinearity=layerNonlinearity,
        switchDefault = 0,
        )

    conv2 = Convolution2DSwitchLayer.Conv2DSwitchLayer(
        networkInput,
        numSwitchOptions = numTasks,
        num_filters = 64,
        filter_size = (4,4),
        stride = (2,2),
        W = lasagne.init.HeUniform(),
        b = lasagne.init.Constant(.1),
        nonlinearity=layerNonlinearity,
        switchDefault = 0,
        )


    conv3 = Convolution2DSwitchLayer.Conv2DSwitchLayer(
        networkInput,
        numSwitchOptions = numTasks,
        num_filters = 64,
        filter_size = (3,3),
        stride = (1,1),
        W = lasagne.init.HeUniform(),
        b = lasagne.init.Constant(.1),
        nonlinearity=layerNonlinearity,
        switchDefault = 0,
        )

    transferLayers.append(conv1)
    transferLayers.append(conv2)
    transferLayers.append(conv3)


    hiddenLayer = lasagne.layers.DenseLayer(
        conv3,
        num_units=512,
        nonlinearity=layerNonlinearity,
        W = lasagne.init.HeUniform(),
        b = lasagne.init.Constant(.1))

    outputLayer = lasagne.layers.DenseLayer(
        hiddenLayer,
        num_units=numOutputs,
        nonlinearity=None,
        W=lasagne.init.HeUniform(),
        b=lasagne.init.Constant(.1))

    return outputLayer, transferLayers