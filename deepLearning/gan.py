
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')
import gpu_config
gpu_config.set_tensorflow([2])


# In[2]:

from keras import backend as K
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt


# In[3]:

sess = tf.Session()
K.set_session(sess)


# In[4]:

D = Sequential()
D.add(Dense(500, input_dim=784))
D.add(Activation("relu"))
D.add(BatchNormalization())

D.add(Dense(500))
D.add(Activation("relu"))
D.add(BatchNormalization())

D.add(Dense(1))
D.add(Activation("sigmoid"))


# In[5]:

G = Sequential()
G.add(Dense(100, input_dim=18))
G.add(Activation("relu"))
G.add(BatchNormalization())

G.add(Dense(100, input_dim=18))
G.add(Activation("relu"))
G.add(BatchNormalization())

G.add(Dense(784))
G.add(Activation("sigmoid"))


# In[6]:

z = tf.placeholder(tf.float32, [None, 18])
x = tf.placeholder(tf.float32, [None, 784])

generated_x = G(z)
d_loss = tf.reduce_mean(tf.log(D(x)) + tf.log(1-D(G(z))))
g_loss = tf.reduce_mean(tf.log(1 - D(G(z))))

d_train = tf.train.AdamOptimizer(0.001).minimize(-d_loss, var_list=D.trainable_weights)
g_train = tf.train.AdamOptimizer(0.001).minimize(g_loss, var_list=G.trainable_weights)

sess.run(tf.global_variables_initializer())


# In[7]:

mnist = input_data.read_data_sets("./MNIST_data",one_hot=True)


# In[8]:

plt.imshow(mnist.train.images[18].reshape(28,28))
plt.gray()


# In[14]:

epoch = 100
batch_size = 100
for i in range(epoch):
    d_losses = []
    g_losses = []
    for j in range(mnist.train.images.shape[0]/batch_size):
        image, label = mnist.train.next_batch(batch_size)
        z_value = np.random.uniform(low=-np.sqrt(3), high= np.sqrt(3), size=[batch_size, 18])
        _, d_loss_val =  sess.run([d_train, d_loss], feed_dict={x:image, z:z_value,K.learning_phase():1})
        z_value = np.random.uniform(low=-np.sqrt(3), high= np.sqrt(3), size=[batch_size, 18])
        _, g_loss_val = sess.run([g_train, g_loss], feed_dict={z:z_value,K.learning_phase():1})
        d_losses.append(d_loss_val)
        g_losses.append(g_loss_val)
    plt.imshow(sess.run(generated_x, feed_dict={z:z_value,K.learning_phase():1})[0].reshape(28,28))
    plt.gray()
    plt.show()
    print "epoch: %d, d_loss: %g. g_loss: %g"


# In[ ]:



