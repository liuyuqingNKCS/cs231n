from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = X.shape[0]
    for i in range(N):

        cur_exp = np.exp(X[i].dot(W))
        loss += -np.log(cur_exp[y[i]]/np.sum(cur_exp))

        # score = X[i].dot(W)
        # loss +=  - score[y[i]] + np.log(np.sum(np.exp(score)))

    loss /= N
    loss += reg*(W**2).sum()


    C = W.shape[1]
    D = W.shape[0]
    for c in range(C):
        for i in range(N):
            score = X[i].dot(W)
            cur_exp = np.exp(score)
            dW[:, c] += X[i]*(cur_exp[c]/cur_exp.sum())  # 1,D
            if y[i] == c:
                dW[:, c] -= X[i]

    dW /= N

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    scores = np.exp(X.dot(W)) # N*C
    loss = (-np.log(scores[np.arange(N), y]/scores.sum(axis=1))).sum() / N + reg*(W**2).sum()
    

    C = W.shape[1]
    D = W.shape[0]
    
    weights = (scores/(scores.sum(axis=1).reshape((N,1)))).reshape((N,C))
    y = y.reshape((N, 1))
    classes = np.array(range(C)).reshape((1,C))
    labels = (y == classes) 
    weights = (weights - labels).reshape((N,C)) 
   
    dW = (X.reshape((N,D,1))*weights.reshape((N,1,C))).sum(axis=0) / N 



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
