#!/usr/bin/env python
# coding: utf-8

# In[2]:


#TUTORIAL LINK: https://youtu.be/wQ8BIBpya2k?t=911
pip install tensorflow


# In[21]:


import tensorflow as tf

mnist = tf.keras.datasets.mnist #28x28 images of handwritten digits 0-9 provided already

(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(len(x_test), len(y_test))


# In[17]:


#scaling / normalizing data so it fits scale 0-1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_train, axis=1)


# In[18]:


#building model
model =tf.keras.models.Sequential()
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


# In[19]:


#train our model
model.fit(x_train,y_train, epochs=3)
##epoch = nomber of passes of training data through algorithm


# In[24]:


#calculating validation loss
##a way to measure the performance of a deep learning model
val_loss, val_acc = model.evaluate(x_test,y_test)
print(val_loss, val_acc)
#test loss
val_loss, val_acc = model.evaluate(x_train,y_train)
print(val_loss, val_acc)
##validation loss should be a bit greater than training loss -> overfitting -> accurate model data and little error


# In[8]:


import matplotlib.pyplot as plt
#tensor
print(x_train[2]) 

#prints out array from dataset from keras module


# In[26]:


#the keras/ number image
plt.imshow(x_train[2])
#we want this in b/w, add the cmp
plt.imshow(x_train[2], cmap = plt.cm.binary)
plt.show()


# In[27]:


#saving the model we traine dinto a new name and using model with new data
model.save("numberModel")
new_model = tf.keras.models.load_model("numberModel")
predictions = new_model.predict([x_test])
print(predictions)


# In[ ]:


#allowing program to predict what number based on model we used


# In[30]:


import numpy as np

print(np.argmax(predictions[11]))


# In[31]:


#how can we see how the prediction see that number they predicted using our model?
#the following code will show the number predicted in a visual format

plt.imshow(x_test[11])
plt.show()


# In[ ]:




