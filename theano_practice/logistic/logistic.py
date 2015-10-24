import theano
import numpy as np
import theano.tensor as T

class Layer(object):
  def __init__(self, in_dim, out_dim, function=None):

    self.rng = np.random.RandomState(1234)
    if function is None:
      function = T.nnet.sigmoid

      self.function = function

      self.W = theano.shared(
          self.rng.uniform(
              low = 0.06,
              high= 0.06,
              size= (in_dim, out_dim)
            ).astype("float32"),
          name="W"
          )

      self.b = theano.shared( 
          np.zeros(out_dim).astype("float32"),
          name="bias"
      )

      self.param = [self.W, self.b]

  def fprop(self, x):
    return self.function(T.dot(x, self.W) + self.b)

if __name__ == "__main__":
   X = T.matrix()
   layer = Layer(in_dim=10, out_dim=3)

   output = layer.fprop(X)

   y = theano.function(
      inputs=[X],
      outputs=output
      )

   x = np.random.rand(15,10).astype("float32")

   print y(x)
