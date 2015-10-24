import theano
import numpy as np
import theano.tensor as T

A = T.matrix()
B = T.matrix()

result = T.dot(A, B)
result2 = T.dot(B,A)
f = theano.function(inputs=[A,B], outputs=[result, result2])

a = np.array([
              [2,-3],
              [4,2]
            ]).astype("float32")

b = np.array([
              [-1,2],
              [3,0]
            ]).astype("float32")

result_list = f(a,b)
for length in range(2):
  print result_list.strip("array")


