# Download datasets

# https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip
# https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip

# Upload datasets to /tmp folder

import os
import zipfile
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

local_zip = 'tmp/rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/')
zip_ref.close()

local_zip = 'tmp/rps-test-set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/')
zip_ref.close()

rock_dir = os.path.join('/tmp/rps/rock')
paper_dir = os.path.join('/tmp/rps/paper')
scissors_dir = os.path.join('/tmp/rps/scissors')

rock_files = os.listdir(rock_dir)
print(rock_files[:10])
paper_files = os.listdir(paper_dir)
print(paper_files[:10])
scissors_files = os.listdir(scissors_dir)
print(scissors_files[:10])

# YOUR TURN

TRAINING_DIR = "/tmp/rps/"
training_datagen = ImageDataGenerator(rescale=1/255.,
                                      rotation_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      horizontal_flip=True)

VALIDATION_DIR = "/tmp/rps-test-set/"
validation_datagen = ImageDataGenerator(rescale=1/255.,
                                        rotation_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        horizontal_flip=True)

train_generator = training_datagen.flow_from_directory(TRAINING_DIR,
                                                       target_size=(150,150),
                                                       batch_size=32,
                                                       class_mode='categorical',
                                                       seed=42)

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              target_size=(150,150),
                                                              batch_size=32,
                                                              class_mode='categorical',
                                                              seed=42)

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    # tf.keras.layers.BatchNormalization(),
    # your code - start
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.4),
    # tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    # tf.keras.layers.Dropout(0.4),
    #your code - end
    tf.keras.layers.Dense(3, activation='softmax') # last layer
])

model.compile(loss = 'categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])
history = model.fit(train_generator, epochs=25, validation_data = validation_generator)

print(history.history['accuracy'])
print(history.history['val_accuracy'])
print(history.history['loss'])
print(history.history['val_loss'])

model.save("results/q2.h5")


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()

