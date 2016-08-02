"""
Author Robert Post

deepmind_rmpsprop code is from Nathan Sprague
from: https://github.com/spragunr/deep_q_rl

which is in turn modified from the Lasagne package:
https://github.com/Lasagne/Lasagne/blob/master/LICENSE
"""


import lasagne
import numpy as np
import theano
import theano.tensor as T
import cPickle
import imp
import TransferLayer
import sys

# sys.path.append("./TransferArchitecture/")

# import DQNNet
# import PolicySwitchNet
# import PolicyPartialSwitchNet
# import FirstRepresentationSwitchNet
# import RepresentationSwitchNet        
# import TaskTransformationNet



def buildTransferNetwork(transferExperimentType, batchSize, inputState, numOutputs, numTasks, convImplementation = "conv", layerNonlinearity = lasagne.nonlinearities.rectify):
    if transferExperimentType == "DQNNet":
        from TransferArchitecture import DQNNet
        architecture = DQNNet

    elif transferExperimentType =="PolicySwitchNet":
        from TransferArchitecture import PolicySwitchNet
        architecture = PolicySwitchNet

    elif transferExperimentType =="PolicyPartialSwitchNet":
        from TransferArchitecture import PolicyPartialSwitchNet
        architecture = PolicyPartialSwitchNet

    elif transferExperimentType =="RepresentationSwitchNet":
        from TransferArchitecture import RepresentationSwitchNet
        architecture = RepresentationSwitchNet

    elif transferExperimentType =="FirstRepresentationSwitchNet":
        from TransferArchitecture import FirstRepresentationSwitchNet
        architecture = FirstRepresentationSwitchNet

    elif transferExperimentType =="TaskTransformationNet":
        from TransferArchitecture import TaskTransformationNet
        architecture = TaskTransformationNet

    return architecture.buildNeuralNetwork(batchSize, inputState, numOutputs, numTasks, convImplementation, layerNonlinearity)


def buildDeepQTransferNetwork_OLD(batchSize, inputState, numOutputs, transferExperimentType, numTasks, convImplementation = "conv", layerNonlinearity = lasagne.nonlinearities.rectify):
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


    numHiddenUnits = 512
    # if transferExperimentType == "fullShare" or transferExperimentType == "representationShare" :
    #     numHiddenUnits /= 8
    # elif transferExperimentType == "layerShare":
    #     numHiddenUnits /= 16

    hiddenLayer = lasagne.layers.DenseLayer(
        conv3,
        num_units=numHiddenUnits,
        nonlinearity=layerNonlinearity,
        W = lasagne.init.HeUniform(),
        b = lasagne.init.Constant(.1))


    if transferExperimentType == "fullShare":
        outputLayer = lasagne.layers.DenseLayer(
            hiddenLayer,
            num_units=numOutputs,
            nonlinearity=None,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1))
    elif transferExperimentType == "layerShare":
        outputLayer = TransferLayer.TransferLayer(
            hiddenLayer,
            num_tasks = numTasks,
            num_units = numOutputs,
            # taskBatchFlag = taskBatchFlag,
            use_shared_layer = True,
            W = lasagne.init.HeUniform(),
            b = lasagne.init.Constant(.1), 
            nonlinearity=None)
        transferLayers.append(outputLayer)
    elif transferExperimentType == "representationShare":
        outputLayer = TransferLayer.TransferLayer(
            hiddenLayer,
            num_tasks = numTasks,
            num_units = numOutputs,
            # taskBatchFlag = taskBatchFlag,
            use_shared_layer = False,
            W = lasagne.init.HeUniform(),
            b = lasagne.init.Constant(.1), 
            nonlinearity=None)
        transferLayers.append(outputLayer)
    else:
        print "Unknown transfer experiment type!\nDefaulting to full share network..."
        outputLayer = lasagne.layers.DenseLayer(
            hiddenLayer,
            num_units=numOutputs,
            nonlinearity=None,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1))    

    return outputLayer, transferLayers


def flipCNNFilters(outputLayer):
        inputLayer = outputLayer

        try:
            imp.find_module('lasagne.layers.dnn')
            dnnFound = True
        except ImportError:
            dnnFound = False

        try:
            imp.find_module('lasagne.layers.cuda_convnet')
            cudaFound = True
        except ImportError:
            cudaFound = False

        while not isinstance(inputLayer, lasagne.layers.input.InputLayer):
            if isinstance(inputLayer, lasagne.layers.conv.Conv2DLayer) or \
            (dnnFound and isinstance(inputLayer, lasagne.layers.dnn.Conv2DDNNLayer)) or \
            (cudaFound and isinstance(inputLayer, lasagne.layers.cuda_convnet.Conv2DCCLayer)):
                inputLayer.W.set_value(inputLayer.W.get_value()[:, :, ::-1, ::-1])

            inputLayer = inputLayer.input_layer


def deepmind_rmsprop(loss_or_grads, params, learning_rate, rho, epsilon):
    """RMSProp updates [1]_.

    Scale learning rates by dividing with the moving average of the root mean
    squared (RMS) gradients.

    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps
    rho : float or symbolic scalar
        Gradient moving average decay factor
    epsilon : float or symbolic scalar
        Small value added for numerical stability

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression

    Notes
    -----
    `rho` should be between 0 and 1. A value of `rho` close to 1 will decay the
    moving average slowly and a value close to 0 will decay the moving average
    fast.

    Using the step size :math:`\\eta` and a decay factor :math:`\\rho` the
    learning rate :math:`\\eta_t` is calculated as:

    .. math::
       r_t &= \\rho r_{t-1} + (1-\\rho)*g^2\\\\
       \\eta_t &= \\frac{\\eta}{\\sqrt{r_t + \\epsilon}}

    References
    ----------
    .. [1] Tieleman, T. and Hinton, G. (2012):
           Neural Networks for Machine Learning, Lecture 6.5 - rmsprop.
           Coursera. http://www.youtube.com/watch?v=O3sxAc4hxZU (formula @5:20)
    """

    grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
    updates = []

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)

        acc_grad = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
        acc_rms = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)

        acc_grad_new = rho * acc_grad + (1 - rho) * grad
        acc_rms_new = rho * acc_rms + (1 - rho) * grad ** 2

        updates.append([acc_grad,  acc_grad_new])
        updates.append([acc_rms, acc_rms_new])
        updates.append([param, (param - learning_rate * (grad / T.sqrt(acc_rms_new - acc_grad_new **2 + epsilon)))])

    return updates

def saveNetworkParams(network, networkFile):
    all_params = [lasagne.layers.helper.get_all_param_values(n) for n in network]
    fileHandle = open(networkFile, 'wb', -1)
    cPickle.dump(all_params, fileHandle)
    fileHandle.close()

def loadNetworkParams(network, paramFile, flipFilters = False):
    paramHandle = open(paramFile, 'rb')
    params = cPickle.load(paramHandle)
    if len(params) != len(network):
        raise Exception("Cant load an array of network parameters into the provided array of qValueFunction approximators - different lengths")
    
    for i in xrange(len(params)):
        lasagne.layers.helper.set_all_param_values(network[i], params[i])
    
    paramHandle.close()

    if flipFilters:
        flipCNNFilters(network)

