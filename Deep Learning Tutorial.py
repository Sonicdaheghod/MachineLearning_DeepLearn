#!/usr/bin/env python
# coding: utf-8

# In[1]:


#TUTORIAL LINK: https://youtu.be/wQ8BIBpya2k?t=451
pip install tensorflow


# In[2]:


import tensorflow as tf

mnist = tf.keras.datasets.mnist #28x28 images of handwritten digits 0-9 provided already

(x_train,y_train),(x_test,y_test) = mnist.load_data()


# In[4]:


import matplotlib.pyplot as plt

print(x_train[2]) 

#prints out array from dataset from keras module


# In[ ]:




