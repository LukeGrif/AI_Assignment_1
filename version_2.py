"""
AI - Project One
Assignment 1
Group Members:
    - Luke Griffin
    - Taha AL-Salihi
    - Patrick Crotty
    - Eoin O'Brien
    - Mark Griffin

Description:
This code implements a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset.
Achieves a classification accuracy of 99% on the training, validation and testing sets.
"""

EPOCHS  = 30      # Training run parameters.  30 epoch run limit.
SPLIT   = 0.2     # 80%/20% train/val split.
SHUFFLE = True    # Shuffle training data before each epoch.
BATCH   = 32      # Minibatch size (note Keras default is 32).
OPT     = 'Adam'  # Adam optimizer.

# Imports

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)                # Initialise system RNG.
import tensorflow
tensorflow.random.set_seed(2)    # and the seed of the Tensorflow backend.
# print(tensorflow.__version__)    # Should be at least 2.0.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense          # Fully-connected layer
from tensorflow.keras.layers import Conv2D         # 2-d Convolutional layer
from tensorflow.keras.layers import MaxPooling2D   # 2-d Max-pooling layer
from tensorflow.keras.layers import Flatten        # Converts 2-d layer to 1-d layer
from tensorflow.keras.layers import Activation     # Nonlinearities
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist


# Extract the training images into "training_inputs"

(training_inputs, training_labels), (testing_inputs, testing_labels) = mnist.load_data()
print(training_inputs.shape, training_inputs.dtype, testing_inputs.shape, testing_inputs.dtype)
training_images = (training_inputs.astype('float32')/255)[:,:,:,np.newaxis]  # Normalised float32 4-tensor.
categorical_training_outputs = to_categorical(training_labels)
testing_images = (testing_inputs.astype('float32')/255)[:,:,:,np.newaxis]
categorical_testing_outputs = to_categorical(testing_labels)

# Model

model = Sequential([
    # First Convolutional Block
    Conv2D(32, kernel_size=(5, 5), padding='same', input_shape=training_images.shape[1:]),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    # Second Convolutional Block
    Conv2D(128, kernel_size=(3, 3), padding='same'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    # Flatten and Fully Connected Layers
    Flatten(),
    Dense(256),
    Activation('relu'),
    Dropout(0.5),
    Dense(10),
    Activation('softmax')
])

# Summary of the model
print("The Keras network model")
model.summary()

# Compile (When we compile, we specify a loss, or error)

model.compile(loss='categorical_crossentropy', optimizer=OPT, metrics=['accuracy'])


# Now we are ready to train.  The routine to do this is "fit".  It tries
# to generate a model that fits the training set.

from tensorflow.keras.callbacks import EarlyStopping

stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, #patience changed
                     verbose=2, mode='auto',
                     restore_best_weights=True)


history = model.fit(training_images, categorical_training_outputs,
                    epochs=EPOCHS,
                    batch_size=BATCH,
                    shuffle=SHUFFLE,
                    validation_split=SPLIT,
                    verbose=2,
                    callbacks=[stop])


# Plot
plt.figure('Training and Validation Losses per epoch', figsize=(8,8))
plt.plot(history.history['loss'],label='training') # Training data error per epoch.
plt.plot(history.history['val_loss'],label='validation') # Validation error per ep.
plt.grid(True)
plt.legend()
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.show()

# We can also test the performance of the network on the completely separate
# testing set.
print("Performance of network on testing set:")
test_loss,test_acc = model.evaluate(testing_images,categorical_testing_outputs)
print("Accuracy on testing data: {:6.2f}%".format(test_acc*100))
print("Test error (loss):        {:8.4f}".format(test_loss))
print("Performance of network:")
print("Accuracy on training data:   {:6.2f}%".format(history.history['accuracy'][-1]*100))
print("Accuracy on validation data: {:6.2f}%".format(history.history['val_accuracy'][-1]*100))
print("Accuracy on testing data:    {:6.2f}%".format(test_acc*100))
