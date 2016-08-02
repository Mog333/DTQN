import numpy as np
import theano.tensor as T
import theano
import lasagne
import DenseSwitchLayer

class DensePartialSwitchLayer():
# class DensePartialSwitchLayer(lasagne.layers.Layer):
    """
    DensePartialSwitchLayer(incoming, numSwitchOptions, numSwitchedUnits, numSharedUnits, switchDefault = 0, 
        W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, **kwargs):

    A layer that has a switch allowing for one of many weight matricies to be used to to compute a function
    The switch can indicate a "task" in which each task does a different computation on the input. 
    
    *The all switchs have the same number of units and same output size.
    *The input layer is automatically flattened (useful for after a convolution layer to get all features)


    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    numSwitchOptions : int
        The number of "dense layers" within this layer - the number of different computations that can be done
    numSwitchedUnits : int
        The number of output units, equal for all switch layers
    W : Theano shared variable, numpy array or callable
        An initializer for the weights of the layer. If a shared variable or a
        numpy array is provided the shape should  be (numSwitchOptions, num_inputs, numSwitchedUnits).
        See :meth:`Layer.create_param` for more information.
    b : Theano shared variable, numpy array, callable or None
        An initializer for the biases of the layer. If a shared variable or a
        numpy array is provided the shape should be (numSwitchedUnits,).
        If None is provided the layer will have no biases.
        See :meth:`Layer.create_param` for more information.
    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    """

    # def __init__(self, incoming, num_tasks, num_units, taskBatchFlag = 0, use_shared_layer = True, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
    def __init__(self, incoming, numSwitchOptions, numSwitchedUnits, numSharedUnits, mergeType = 'sum', switchDefault = 0, W1=lasagne.init.GlorotUniform(), W2=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        # super(DensePartialSwitchLayer, self).__init__(incoming, **kwargs)
        # super(DensePartialSwitchLayer, self).__init__(incoming, **kwargs)

        self.nonlinearity       = (lasagne.nonlinearities.identity if nonlinearity is None else nonlinearity)
        self.numSwitchOptions   = numSwitchOptions
        self.numSwitchedUnits   = numSwitchedUnits
        self.numSharedUnits     = numSharedUnits
        # self.batchSize          = self.input_shape[0]
        self.currentSwitchIndex = theano.shared(switchDefault)
        self.mergeType          = mergeType


        if mergeType != 'sum' and mergeType != 'max' and mergeType != 'concat':
            raise Exception("merge type must be one of {sum|max|concat}")
            return

        if mergeType == 'concat':
            self.numOutputUnits = self.numSwitchedUnits + self.numSharedUnits
        else:
            if numSwitchedUnits != numSharedUnits:
                raise Exception("When doing a merge other than concat the number of switched units must equal the number of shared units - the passed parameters don't match")
            self.numOutputUnits = self.numSwitchedUnits


        # self.num_inputs = int(np.prod(self.input_shape[1:]))


        self.switchedLayer = DenseSwitchLayer.DenseSwitchLayer(
            incoming, 
            self.numSwitchOptions, 
            self.numSwitchedUnits,
            switchDefault = switchDefault, 
            nonlinearity = None, 
            W = W1, 
            b = None)


        self.sharedLayer = lasagne.layers.DenseLayer(
            incoming,
            num_units=self.numSharedUnits,
            nonlinearity=None,
            W = W2,
            b = None,
        )


        self.incomings = [self.switchedLayer, self.sharedLayer]


        if self.mergeType == 'concat':
            self.mergedLayers = lasagne.layers.ConcatLayer(self.incomings, axis=1)

        elif self.mergeType == 'max':
            self.mergedLayers = lasagne.layers.ElemwiseMergeLayer(self.incomings, theano.tensor.maximum)

        elif self.mergeType == 'sum':
            self.mergedLayers = lasagne.layers.ElemwiseMergeLayer(self.incomings, theano.tensor.add)
        else:
            raise Exception("Shouldnt get here. we checked merge type upon initialization")

        if b is None:
            # self.b = None
            biasedOut = self.mergedLayers

        else:
            biasedOut = lasagne.layers.BiasLayer(self.mergedLayers, b)

            # self.b = self.add_param(b, (self.numOutputUnits,), name="b", regularizable=False)

        self.lastLayer = lasagne.layers.NonlinearityLayer(biasedOut, nonlinearity)



    # def get_output_shape_for(self, input_shape):
    #     return (input_shape[0], self.numOutputUnits)

    # def get_output_for(self, input, **kwargs):
    #     print "\n\n"+str(type(input))+"\n\n"
    #     if input.ndim > 2:
    #         #Flatten the input tensor into a batch of feature vectors
    #         input = input.flatten(2)

    #     ## switchedOutput = self.switchedLayer.get_output_for(input)
    #     ## sharedOutput = self.sharedLayer.get_output_for(input)
    #     ## incomings = (switchedOutput, sharedOutput)

    #     ## activation = T.dot(input, self.W[self.currentSwitchIndex])
        
    #     ## activation = self.mergedLayers.get_output_for(input)
        
    #     activation = lasagne.layers.get_output(self.mergedLayers, input)
    #     print "\n\n"+str(type(activation))+"\n\n"

    #     # activation = mergedLayers
    #     if self.b is not None:
    #         activation = activation + self.b.dimshuffle('x', 0)

    #     return self.nonlinearity(activation)


    def setSwitchIndex(self, newSwitchIndex):
        if newSwitchIndex < 0 or newSwitchIndex > self.numSwitchOptions - 1:
            raise Exception("Cant set switch - value out of bounds!")

        self.currentSwitchIndex.set_value(newSwitchIndex)
        
        self.switchedLayer.setSwitchIndex(newSwitchIndex)

        return self.currentSwitchIndex.get_value()

    def getSwitchIndex(self):
        return self.currentSwitchIndex.get_value()



def test():
    batchSize   = 2
    inputWidth = 3
    inputHeight = 3
    inputLength = 3
    numSwitchOptions = 2
    numSwitchedUnits = 2
    numSharedUnits = 2
    switchDefault = 0
    stateShape = (batchSize, 1, inputHeight, inputWidth)
    mergeType= 'concat'

    networkInput = lasagne.layers.InputLayer(shape=stateShape)
    # def __init__(self, incoming, numSwitchOptions, numSwitchedUnits, numSharedUnits, switchDefault = 0, mergeType = 'sum', W1=lasagne.init.GlorotUniform(), W2=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, **kwargs):

    convFunction = lasagne.layers.conv.Conv2DLayer
    conv1 = convFunction(
            networkInput, 
            num_filters = 1, 
            filter_size = (2,2), 
            stride=(1,1),  
            W = lasagne.init.HeUniform(),
            b = lasagne.init.Constant(.1))

    # hiddenLayer = lasagne.layers.DenseLayer(
    #     networkInput,
    #     num_units=4,
    #     W = lasagne.init.HeUniform(),
    #     b = lasagne.init.Constant(.1))



    dpsl = DensePartialSwitchLayer(conv1, numSwitchOptions, numSwitchedUnits, numSharedUnits, mergeType = mergeType, switchDefault = switchDefault, nonlinearity=lasagne.nonlinearities.identity, W1 = lasagne.init.HeUniform(), W2 = lasagne.init.HeUniform(), b = lasagne.init.Constant(.1))

    dpsl.switchedLayer.W.set_value(np.random.randint(2, size=dpsl.switchedLayer.W.get_value().shape))
    dpsl.sharedLayer.W.set_value(np.random.randint(2, size=dpsl.sharedLayer.W.get_value().shape))

    print "\nNetwork Weights:\nSwitched:"
    print dpsl.switchedLayer.W.get_value()
    print "\nShared:"
    print dpsl.sharedLayer.W.get_value()


    inputTensor = T.tensor4('state')
    inputValues = np.random.random(stateShape)
    inputState = theano.shared(inputValues)
    print "\nInput values: \n"
    print inputValues


    output = lasagne.layers.get_output(dpsl.lastLayer, inputTensor)
    output2 = lasagne.layers.get_output(conv1, inputTensor)

    f = theano.function([], [output, output2], givens = {inputTensor:inputState}, on_unused_input='warn')

    
    print "\nFunction results:\n"
    dpsl.setSwitchIndex(0)
    print f()
    print "\nfunction results 2:\n"
    dpsl.setSwitchIndex(1)
    print f()


if __name__ == "__main__":
    test()