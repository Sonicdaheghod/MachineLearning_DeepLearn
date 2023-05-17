#!/usr/bin/env python
# coding: utf-8

# In[15]:


#TUTORIAL LINK: https://youtu.be/wQ8BIBpya2k?t=911
pip install tensorflow


# In[16]:


import tensorflow as tf

mnist = tf.keras.datasets.mnist #28x28 images of handwritten digits 0-9 provided already

(x_train,y_train),(x_test,y_test) = mnist.load_data()


# In[26]:


#scaling / normalizing data so it fits scale 0-1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_train, axis=1)


# In[29]:


#building model
model  =tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
#hidden layer
model.add(tf.keras.layers.Dense(128,activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation = tf.nn.relu))
##probability distribution- softmax 
model.add(tf.keras.layers.Dense(10,activation = tf.nn.softmax))

##loss= degree of error
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=["accuracy"])


# In[30]:


#train our model
model.fit(x_train,y_train, epochs=2)
##epoch = nomber of passes of training data through algorithm


# In[23]:


import matplotlib.pyplot as plt
#tensor
print(x_train[2]) 

#prints out array from dataset from keras module


# In[31]:


#the keras/ number image
plt.imshow(x_train[2])
#we want this in b/w, add the cmp
plt.imshow(x_train[2], cmap = plt.cm.binary)
plt.show()


# In[ ]:




