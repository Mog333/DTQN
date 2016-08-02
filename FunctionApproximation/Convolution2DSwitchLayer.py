import numpy as np
import theano.tensor as T
import theano
import lasagne


def canUseDNN():
    try:
        from theano.sandbox.cuda import dnn


        if not theano.sandbox.cuda.cuda_enabled:
            raise ImportError(
                    "requires GPU support -- see http://lasagne.readthedocs.org/en/"
                    "latest/user/installation.html#gpu-support")  # pragma: no cover
        elif not dnn.dnn_available():
            raise ImportError(
                    "cuDNN not available: %s\nSee http://lasagne.readthedocs.org/en/"
                    "latest/user/installation.html#cudnn" %
                    dnn.dnn_available.msg)  # pragma: no cover


        # convolution = dnn.dnn_conv
        print "Using cuDNN convolution function: theano.sandbox.cuda.dnn.dnn_conv"
        return True

    except Exception as e:
        print "Using default convolution function: theano.tensor.nnet.conv2d"
        return False
        # convolution = theano.tensor.nnet.conv2d



class Conv2DSwitchLayer(lasagne.layers.Layer):
    """
    lasagne.layers.Conv2DSwitchLayer(incoming, num_filters, filter_size,
    stride=1, pad=0, untie_biases=False,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True,
    n=None, **kwargs)
    Convolutional layer base class
    Base class for performing an `n`-dimensional convolution on its input,
    optionally adding a bias and applying an elementwise nonlinearity. Note
    that this class cannot be used in a Lasagne network, only its subclasses
    can (e.g., :class:`Conv1DLayer`, :class:`Conv2DLayer`).
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. Must
        be a tensor of 2+`n` dimensions:
        ``(batch_size, num_input_channels, <n spatial dimensions>)``.
    num_filters : int
        The number of learnable convolutional filters this layer has.
    filter_size : int or iterable of int
        An integer or an `n`-element tuple specifying the size of the filters.
    stride : int or iterable of int
        An integer or an `n`-element tuple specifying the stride of the
        convolution operation.
    pad : int, iterable of int, 'full', 'same' or 'valid' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.
        A single integer results in symmetric zero-padding of the given size on
        all borders, a tuple of `n` integers allows different symmetric padding
        per dimension.
        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.
        ``'same'`` pads with half the filter size (rounded down) on both sides.
        When ``stride=1`` this results in an output size equal to the input
        size. Even filter size is not supported.
        ``'valid'`` is an alias for ``0`` (no padding / a valid convolution).
        Note that ``'full'`` and ``'same'`` can be faster than equivalent
        integer values due to optimizations by Theano.
    untie_biases : bool (default: False)
        If ``False``, the layer will have a bias parameter for each channel,
        which is shared across all positions in this channel. As a result, the
        `b` attribute will be a vector (1D).
        If ``True``, the layer will have separate bias parameters for each
        position in each channel. As a result, the `b` attribute will be an
        `n`-dimensional tensor.
    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a tensor of 2+`n` dimensions with shape
        ``(num_filters, num_input_channels, <n spatial dimensions>)``.
        See :func:`lasagne.utils.create_param` for more information.
    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_filters,)`` if `untied_biases` is set to
        ``False``. If it is set to ``True``, its shape should be
        ``(num_filters, <n spatial dimensions>)`` instead.
        See :func:`lasagne.utils.create_param` for more information.
    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.
    flip_filters : bool (default: True)
        Whether to flip the filters before sliding them over the input,
        performing a convolution (this is the default), or not to flip them and
        perform a correlation. Note that for some other convolutional layers in
        Lasagne, flipping incurs an overhead and is disabled by default --
        check the documentation when using learned weights from another layer.
    n : int or None
        The dimensionality of the convolution (i.e., the number of spatial
        dimensions of each feature map and each convolutional filter). If
        ``None``, will be inferred from the input shape.
    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.
    Attributes
    ----------
    W : Theano shared variable or expression
        Variable or expression representing the filter weights.
    b : Theano shared variable or expression
        Variable or expression representing the biases.
    """


    def __init__(self, incoming, numSwitchOptions, num_filters, filter_size, stride=(1, 1), 
                 pad=0,
                 untie_biases=False,
                 W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
                 nonlinearity=lasagne.nonlinearities.rectify, 
                 flip_filters=True,         
                 n=2, switchDefault = 0, **kwargs):
        # Convolution=T.nnet.conv.conv2d,
        # super(BaseConvLayer, self).__init__(incoming, **kwargs)
        super(Conv2DSwitchLayer, self).__init__(incoming, **kwargs)

        # self.convolution = convolution

        self.numSwitchOptions = numSwitchOptions
        self.currentSwitchIndex = theano.shared(switchDefault)


        if nonlinearity is None:
            self.nonlinearity = lasagne.nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        if n is None:
            n = len(self.input_shape) - 2
        elif n != len(self.input_shape) - 2:
            raise ValueError("Tried to create a %dD convolution layer with "
                             "input shape %r. Expected %d input dimensions "
                             "(batchsize, channels, %d spatial dimensions)." %
                             (n, self.input_shape, n+2, n))
        self.n = n
        self.num_filters = num_filters
        self.filter_size = lasagne.utils.as_tuple(filter_size, n, int)
        self.flip_filters = flip_filters
        self.stride = lasagne.utils.as_tuple(stride, n, int)
        self.untie_biases = untie_biases

        if pad == 'same':
            if any(s % 2 == 0 for s in self.filter_size):
                raise NotImplementedError(
                    '`same` padding requires odd filter size.')
        if pad == 'valid':
            self.pad = lasagne.utils.as_tuple(0, n)
        elif pad in ('full', 'same'):
            self.pad = pad
        else:
            self.pad = lasagne.utils.as_tuple(pad, n, int)

        self.switchWShape = (self.numSwitchOptions,) +self.get_W_shape()

        self.W = self.add_param(W, self.switchWShape, name="W")
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_filters,) + self.output_shape[2:]
            else:
                biases_shape = (num_filters,)
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)

    def get_W_shape(self):
        """Get the shape of the weight matrix `W`.
        Returns
        -------
        tuple of int
            The shape of the weight matrix.
        """
        num_input_channels = self.input_shape[1]
        return (self.num_filters, num_input_channels) + self.filter_size

    def get_output_shape_for(self, input_shape):
        pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * self.n
        batchsize = input_shape[0]
        return ((batchsize, self.num_filters) +
                tuple(lasagne.layers.conv.conv_output_length(input, filter, stride, p)
                      for input, filter, stride, p
                      in zip(input_shape[2:], self.filter_size,
                             self.stride, pad)))

    def get_output_for(self, input, **kwargs):
        conved = self.convolve(input, **kwargs)

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            activation = conved + T.shape_padleft(self.b, 1)
        else:
            activation = conved + self.b.dimshuffle(('x', 0) + ('x',) * self.n)

        return self.nonlinearity(activation)


    def convolve(self, input, **kwargs):
        # border_mode = 'half' if self.pad == 'same' else self.pad
        border_mode = 'valid'

        if canUseDNN():
            from theano.sandbox.cuda.dnn import dnn_conv

            conved = dnn_conv(input, kerns = self.W[self.currentSwitchIndex],
                                  subsample=self.stride,
                                  border_mode=border_mode)
        else:
            conved = theano.tensor.nnet.conv2d(input, self.W[self.currentSwitchIndex],
                                  self.input_shape,
                                  self.get_W_shape(),
                                  subsample=self.stride,
                                  border_mode=border_mode)

        # conved = theano.tensor.nnet.conv2d(input, self.W[self.currentSwitchIndex],

                                  # filter_flip=self.flip_filters)
        return conved






    def setSwitchIndex(self, newSwitchIndex):
        if newSwitchIndex < 0 or newSwitchIndex > self.numSwitchOptions - 1:
            raise Exception("Cant set switch - value out of bounds!")

        self.currentSwitchIndex.set_value(newSwitchIndex)
        return self.currentSwitchIndex.get_value()

    def getSwitchIndex(self):
        return self.currentSwitchIndex.get_value()











def test():
    np.random.seed(5)
    batchSize   = 2
    numChannels = 1
    inputWidth = 3
    inputHeight = 3
    numSwitchOptions = 2
    numSwitchedUnits = 2
    inputShape = shape=(batchSize, numChannels, inputHeight, inputWidth)

    networkInput = lasagne.layers.InputLayer(inputShape)


    convLayer = Conv2DSwitchLayer(networkInput, numSwitchOptions, 
        num_filters= 2, 
        filter_size=(2,2), 
        stride = (1,1), 
        nonlinearity = None, 
        W = lasagne.init.HeUniform(),
        b = None)

    print convLayer.switchWShape
    filterData = (np.random.randint(4, size=convLayer.switchWShape).astype(theano.config.floatX) + 1) / 2.0

    convLayer.W.set_value(filterData)

    inputTensor = T.tensor4('state')
    output = lasagne.layers.get_output(convLayer, inputTensor)
    
    inputValues = np.random.random(inputShape).astype(theano.config.floatX)
    inputState = theano.shared(inputValues)
    print "Input values: \n"
    print inputValues

    f = theano.function([], output, givens = {inputTensor:inputState})

    print "\nNetwork Weights:\n"
    print convLayer.W.get_value()

    print "\nFunction results:\n"
    convLayer.setSwitchIndex(0)
    print np.asarray( f() )
    print "\nfunction results 2:\n"
    convLayer.setSwitchIndex(1)
    print np.asarray( f() )


if __name__ == "__main__":
    test()