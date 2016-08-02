import numpy as np
import theano.tensor as T
import theano
import lasagne

class TransferLayer(lasagne.layers.Layer):
    """
    TransferLayer(incoming, num_tasks, num_units, useSharedLayer = True
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, **kwargs)


    A layer consisting of DenseLayers, one for each task, 
    and an optionally single shared layer. Each output layer has the same size
    Given which task its on the output is the sum of that tasks dense layer and the shared layer

    EX 2 tasks with 3 outputs will have 2 (3 with a shared layer) dense layers of shape (incoming, num_units)

    !!! The output is still a single dense layer with shape (batch (inputshape[0]), num_units) !!!
    The output is dependand on which tasks is currently active.
    The output is calculated from the sum of the task output layer and the shared layer

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    num_units : int
        The number of units of the layer
    num_tasks : int
        The number of tasks, each task each task is another layer
    use_shared_layer : bool
        flag for usage of a shared weight layer
    W : Theano shared variable, numpy array or callable
        An initializer for the weights of the layer. If a shared variable or a
        numpy array is provided the shape should  be (num_inputs, num_units).
        See :meth:`Layer.create_param` for more information.
    b : Theano shared variable, numpy array, callable or None
        An initializer for the biases of the layer. If a shared variable or a
        numpy array is provided the shape should be (num_units,).
        If None is provided the layer will have no biases.
        See :meth:`Layer.create_param` for more information.
    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.
    Examples
    --------
    >>> import TransferLayer
    >>> l_in = InputLayer((100, 20))
    >>> l1 = DenseLayer(l_in, num_tasks = 3, num_units=50, use_shared_layer = True)
    Notes
    -----
    If the input to this layer has more than two axes, it will flatten the
    trailing axes. This is useful for when a dense layer follows a
    convolutional layer, for example. It is not necessary to insert a
    :class:`FlattenLayer` in this case.
    """

    # def __init__(self, incoming, num_tasks, num_units, taskBatchFlag = 0, use_shared_layer = True, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
    def __init__(self, incoming, num_tasks, num_units, use_shared_layer = True, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        super(TransferLayer, self).__init__(incoming, **kwargs)

        self.nonlinearity     = (lasagne.nonlinearities.identity if nonlinearity is None else nonlinearity)
        self.num_tasks        = num_tasks
        self.num_units        = num_units
        self.use_shared_layer = use_shared_layer
        self.batchSize        = self.input_shape[0]
        # self.taskBatchFlag    = taskBatchFlag

        # assert self.taskBatchFlag >= 0

        # self.taskIndices = theano.shared(np.zeros(self.batchSize, dtype='int32'))
        self.currentTaskIndex = theano.shared(np.zeros(1, dtype='int32'))

        num_inputs = int(np.prod(self.input_shape[1:]))

        self.W = self.add_param(W, (num_tasks, num_inputs, num_units), name="W")


        


        if use_shared_layer:
            self.W_Shared = self.add_param(W, (num_inputs, num_units), name="W_Shared")
        else:
            self.W_Shared = None

        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b", regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        if self.W_Shared is not None:
            # if self.taskBatchFlag == 0:
            #     activation = T.batched_dot(input,self.W[self.taskIndices] + [self.W_Shared] * self.batchSize)
            # elif self.taskBatchFlag > 0:
            #     activation = T.dot(input, self.W[self.taskIndices[0]] + self.W_Shared)

            activation = T.dot(input, self.W[self.currentTaskIndex] + self.W_Shared)
        else:
            if self.num_tasks == 1:
                #Using transferlayer as normal dense layer
                activation = T.dot(input,self.W[0])
            else:
                # if self.taskBatchFlag == 0:
                #     activation = T.batched_dot(input,self.W[self.taskIndices])
                # elif self.taskBatchFlag > 0:
                    # activation = T.dot(input, self.W[self.taskIndices[0]])
                activation = T.dot(input, self.W[self.currentTaskIndex])

        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)

    def setTaskIndex(self, newTaskIndex):
        self.currentTaskIndex.set_value(newTaskIndex)
        return self.currentTaskIndex.get_value()

    def getTaskIndex(self):
        return self.currentTaskIndex.get_value()


    # def setTaskIndices(self, newIndexArray):
    #     self.taskIndices.set_value(newIndexArray)
    #     return self.taskIndices.get_value()

    # def getTaskIndices(self):
    #     return self.taskIndices.get_value()

