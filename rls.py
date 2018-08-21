import numpy as np
import theano
import theano.tensor as T

class RLS:
  def __init__(self, order, gamma):
    weight_value = np.zeros(order,
                           dtype = theano.config.floatX)
    P_value = 1.0/gamma*np.eye(order,
                               dtype=theano.config.floatX)
    weight = theano.shared(value = weight_value,
                           name = 'weight',
                           borrow = True)
    P = theano.shared(value = P_value,
                      name='P',
                      borrow=True)
        
    input = T.vector("input")
    target = T.dscalar('target')
    lmbda = T.dscalar('lmbda')
    
    output = T.dot(input, weight)
    error = target - output
    xP = T.dot(P.T,input) # input.T*P (vector)
    a = 1.0/(lmbda + T.dot(xP, input)) # scalar
    g = T.dot(P, input)*a # P*input*a (vector)
    new_weight = weight + error*g
    newP = (1.0/lmbda) * (P - T.outer(g,xP))
    
    self.observe = theano.function(inputs = [input, target, lmbda],
                                   outputs = [output, error],
                                   updates = [
                                      (weight, new_weight),
                                      (P, newP)
                                   ])
