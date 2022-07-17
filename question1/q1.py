# In the case of creating neural networks, the sample I like to use is one where it learns the relationship between two numbers.
# You should train a neural network to do the equivalent task by feeding it with a set of Xs, and a set of Ys
# It should be able to figure out the relationship between them.

import tensorflow as tf
import numpy as np
from tensorflow import keras

# Define and Compile the Neural Network

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=[1]),
    tf.keras.layers.Dense(1, activation='relu')])
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')

# Next up we'll feed in some data. In this case we are taking 6 xs and 6ys.
# You can see that the relationship between these is that y=2x-1, so where x = -1, y=-3 etc.

xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# The process of training the neural network, where it 'learns' the relationship between the Xs and Ys is in the model.fit call.
# This is where it will go through the loop we spoke about above, making a guess, measuring how good or bad it is (aka the loss),
# using the opimizer to make another guess etc. It will do it for the number of epochs you specify. When you run this code,
# you'll see the loss on the right hand side.

model.fit(np.expand_dims(xs, axis=-1), ys, epochs=500)
print(model.predict([10.0]))

model.save('results/q1.h5')
