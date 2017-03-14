
# coding: utf-8

# In[20]:

import gpu_config
gpu_config.set_tensorflow([0])


# In[21]:

get_ipython().magic('matplotlib inline')
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Convolution2D
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report


# In[22]:

ntt_data = pd.read_csv('/home/d-hacks/data/compartment_line/ntt_dataset.csv')
labels = np.load('/home/d-hacks/data/compartment_line/data_set1/npy/label_car_final.npy')


# In[23]:

y_train_drgn = ntt_data.Seikai.as_matrix().reshape(11234,1)
y_train_ntt = ntt_data.Suitei.as_matrix().reshape(11234,1)
x_train = []
for i in ntt_data.ImageId:
    x_train.append(io.imread('/home/d-hacks/data/compartment_line/data_set1/src_img/picture1/image%05d.jpg' % (i+1), as_grey=True).reshape(224, 224, 1))

x_train = np.array(x_train).astype(np.float32)


# In[24]:

x_train, x_test, y_train_drgn, y_test_drgn, y_train_ntt, y_test_ntt = train_test_split(x_train,y_train_drgn, y_train_ntt, test_size=1234./11234)
x_validation, x_test, y_drgn_validation, y_drgn_test, y_ntt_validation, y_ntt_test = train_test_split(x_test, y_test_drgn, y_test_ntt, test_size=1000./1234)
print x_train.shape, x_validation.shape, x_test.shape


# In[37]:

model = Sequential()
model.add(Convolution2D(32, 3, 3,  border_mode='same', input_shape=(224, 224, 1)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Convolution2D(32, 3, 3, border_mode='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Convolution2D(16, 3, 3, border_mode='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Convolution2D(16, 3, 3, border_mode='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(50))
model.add(Activation("relu"))
model.add(Dense(1))

model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])


# In[38]:

model.summary()


# In[39]:

#drgnman
#model.fit(x_train, y_train_drgn, nb_epoch=15, batch_size=32, validation_data=[x_validation, y_drgn_validation], validation_split=0.2)
#NTT
model.fit(x_train, y_train_ntt, nb_epoch=15, batch_size=32, validation_data=[x_validation, y_ntt_validation], validation_split=0.2)


# In[40]:

evaluation = model.evaluate(x_test, y_drgn_test, batch_size=16)
predict = model.predict_classes(x_test, batch_size=16)
print 'Evaluation', evaluation
print 'Accuracy', accuracy_score(y_drgn_test, predict)
print 'Recall', recall_score(y_drgn_test, predict)
print 'f1', f1_score(y_drgn_test, predict)
print 'Precision', precision_score(y_drgn_test, predict)
print 'ROC AUC', roc_auc_score(y_drgn_test, predict)


# In[ ]:




# In[35]:




# In[35]:




# In[ ]:



