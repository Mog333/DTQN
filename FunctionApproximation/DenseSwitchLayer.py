import numpy as np
import theano.tensor as T
import theano
import lasagne

class DenseSwitchLayer(lasagne.layers.Layer):
    """
    DenseSwitchLayer(incoming, numSwitchOptions, numSwitchedUnits, switchDefault = 0, 
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
    def __init__(self, incoming, numSwitchOptions, numSwitchedUnits, switchDefault = 0, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        super(DenseSwitchLayer, self).__init__(incoming, **kwargs)


        self.nonlinearity     = (lasagne.nonlinearities.identity if nonlinearity is None else nonlinearity)
        self.numSwitchOptions = numSwitchOptions
        self.numSwitchedUnits = numSwitchedUnits
        self.batchSize        = self.input_shape[0]
        self.currentSwitchIndex = theano.shared(switchDefault)

        self.num_inputs = int(np.prod(self.input_shape[1:]))
        self.W = self.add_param(W, (self.numSwitchOptions, self.num_inputs, self.numSwitchedUnits), name="W")

        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (self.numSwitchedUnits,), name="b", regularizable=False)


    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.numSwitchedUnits)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            #Flatten the input tensor into a batch of feature vectors
            input = input.flatten(2)

        activation = T.dot(input, self.W[self.currentSwitchIndex])

        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)


    def setSwitchIndex(self, newSwitchIndex):
        if newSwitchIndex < 0 or newSwitchIndex > self.numSwitchOptions - 1:
            raise Exception("Cant set switch - value out of bounds!")

        self.currentSwitchIndex.set_value(newSwitchIndex)
        return self.currentSwitchIndex.get_value()

    def getSwitchIndex(self):
        return self.currentSwitchIndex.get_value()



def test():
    batchSize   = 2
    inputLength = 3
    numSwitchOptions = 2
    numSwitchedUnits = 2

    networkInput = lasagne.layers.InputLayer(shape=(batchSize, inputLength))
    d1 = DenseSwitchLayer(networkInput, numSwitchOptions, numSwitchedUnits, switchDefault = 0, nonlinearity=lasagne.nonlinearities.identity, W = lasagne.init.HeUniform(), b = lasagne.init.Constant(.1))

    d1.W.set_value(np.random.randint(2, size=(numSwitchOptions, d1.num_inputs, numSwitchedUnits)))

    inputTensor = T.matrix('state')
    output = lasagne.layers.get_output(d1, inputTensor)
    
    inputValues = np.random.random((batchSize, inputLength))
    inputState = theano.shared(inputValues)
    print "Input values: \n"
    print inputValues

    f = theano.function([], output, givens = {inputTensor:inputState})

    print "\nNetwork Weights:\n"
    print d1.W.get_value()

    print "\nFunction results:\n"
    d1.setSwitchIndex(0)
    print f()
    print "\nfunction results 2:\n"
    d1.setSwitchIndex(1)
    print f()


if __name__ == "__main__":
    test()