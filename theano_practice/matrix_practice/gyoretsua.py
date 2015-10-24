import theano
import numpy as np
import theano.tensor as T

A = T.matrix()
B = T.matrix()

result = A + B
f = theano.function(inputs=[A,B], outputs=result)

a = np.array([
              [2,-3],
              [4,2]
            ]).astype("float32")

b = np.array([
              [-1,2],
              [3,0]
            ]).astype("float32")


print f(a,b)  
