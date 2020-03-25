from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.params['W1'] = weight_scale * \
            np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = 0
        self.params['W2'] = weight_scale * \
            np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = 0
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out_1, cache_1 = affine_forward(
            X, self.params["W1"], self.params["b1"])
        out_2, cache_2 = relu_forward(out_1)
        out_3, cache_3 = affine_forward(
            out_2, self.params["W2"], self.params["b2"])
        scores = out_3

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dsoftmax_loss = softmax_loss(scores, y)
        loss += self.reg / 2 * \
            ((self.params["W1"]**2).sum() + (self.params["W2"]**2).sum())
        dlast, dW2, db2 = affine_backward(dsoftmax_loss, cache_3)
        dout_2 = relu_backward(dlast, cache_2)
        dout_1, dW1, db1 = affine_backward(dout_2, cache_1)
        dW1 += self.reg * self.params["W1"]
        dW2 += self.reg * self.params["W2"]
        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

def affine_batch_relu_forward(x, w, b, gamma, beta, bn_param):
    out, fc_cache = affine_forward(x, w, b)
    out, batch_cache = batchnorm_forward(out, gamma, beta, bn_param)
    out, relu_cache = relu_forward(out)
    cache = (fc_cache, batch_cache, relu_cache)
    return out, cache

def affine_batch_relu_backward(dout, cache):
    dout = relu_backward(dout, cache[2])
    dout, dgamma, dbeta = batchnorm_backward_alt(dout, cache[1])
    dx, dw, db = affine_backward(dout, cache[0])
    return dx, dw, db, dgamma, dbeta
class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self,
                 hidden_dims,
                 input_dim=3 * 32 * 32,
                 num_classes=10,
                 dropout=1,
                 normalization=None,
                 reg=0.0,
                 weight_scale=1e-2,
                 dtype=np.float32,
                 seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dims = [input_dim] + hidden_dims + [num_classes]
        for index, value in enumerate(dims):
            if index != 0:
                keys = ["W{}".format(index), "b{}".format(index)]
                self.params[keys[0]] = weight_scale * \
                    np.random.randn(dims[index-1], value)
                self.params[keys[1]] = np.zeros((dims[index]))
                if self.normalization == "batchnorm":
                    if index != len(dims) -1 :
                        keys.extend(["gamma{}".format(index), "beta{}".format(index)])
                        self.params[keys[2]] = np.ones((dims[index]))
                        self.params[keys[3]] = np.zeros((dims[index]))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == 'batchnorm':
            self.bn_params = [{'mode': 'train'}
                              for i in range(self.num_layers - 1)]
        if self.normalization == 'layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization == 'batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

        if self.normalization == "batchnorm":
            cur_out = X
            out = []
            cache = []
            for index in range(self.num_layers):
                
                if index != self.num_layers - 1:
                    cur_out, cur_cache = affine_batch_relu_forward(cur_out, self.params["W{}".format(index+1)], self.params["b{}".format(index+1)], self.params["gamma{}".format(index+1)], self.params["beta{}".format(index+1)], self.bn_params[index])
                    out.append(cur_out)
                    cache.append(cur_cache)
                    if self.use_dropout:
                        cur_out, cur_cache = dropout_forward(out[-1], self.dropout_param)
                        out.append(cur_out)
                        cache.append(cur_cache)
                else:
                    cur_out, cur_cache = affine_forward(cur_out, self.params["W{}".format(index+1)], self.params["b{}".format(index+1)])
                    out.append(cur_out)
                    cache.append(cur_cache)
            scores = out[-1]
        else:
            cur_out = X
            out = []
            cache = []
            for index in range(self.num_layers):
                cur_out, cur_cache = affine_forward(
                    cur_out, self.params["W{}".format(index+1)], self.params["b{}".format(index+1)])
                out.append(cur_out)
                cache.append(cur_cache)
                if index != self.num_layers - 1:
                    cur_out, cur_cache = relu_forward(out[-1])
                    out.append(cur_out)
                    cache.append(cur_cache)
                    if self.use_dropout:
                        cur_out, cur_cache = dropout_forward(out[-1], self.dropout_param)
                        out.append(cur_out)
                        cache.append(cur_cache)
            scores = out[-1]


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dsoftmax_loss = softmax_loss(scores, y)
        loss += self.reg / 2 * sum(
            [(self.params["W{}".format(index+1)]**2).sum() for index in range(self.num_layers)])
        
        cache_index = len(cache) - 1
        dlast, grads["W{}".format(self.num_layers)], grads["b{}".format(self.num_layers)] = affine_backward(dsoftmax_loss, cache[cache_index])
        grads["W{}".format(self.num_layers)] += self.reg * self.params["W{}".format(self.num_layers)] * 2 / 2
        cache_index -= 1
        dout = dlast
        if self.normalization == "batchnorm":
            for i in reversed(range(self.num_layers)):
                if i >= 1:
                    if self.use_dropout:
                        dout = dropout_backward(dout, cache[cache_index])
                        cache_index -= 1
                    dout, grads["W{}".format(i)], grads["b{}".format(i)], grads["gamma{}".format(i)], grads["beta{}".format(i)]   = affine_batch_relu_backward(dout, cache[cache_index])
                    grads["W{}".format(i)] += self.reg * self.params["W{}".format(i)] * 2 / 2
                    cache_index -= 1
        else:
            for i in reversed(range(self.num_layers)):
                if i >= 1:
                    if self.use_dropout:
                        dout = dropout_backward(dout, cache[cache_index])
                        cache_index -= 1
                    dout = relu_backward(dout, cache[cache_index])
                    cache_index -= 1
                    dout, grads["W{}".format(i)], grads["b{}".format(i)] = affine_backward(dout, cache[cache_index])
                    grads["W{}".format(i)] += self.reg * self.params["W{}".format(i)] * 2 / 2
                    cache_index -= 1


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
