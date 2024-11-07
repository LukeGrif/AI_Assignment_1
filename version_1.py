# kerasMNIST_ex_CNN.ipynb
# 
# First pass at a CNN for MNIST.
#  --
#  
# Here we use one 3 x 3 x 10 convolutional layer followed by a 2 x 2 
# maxpooling layer.  The net is then flattened to 1-d and a standard
# (ReLU) hidden layer is applied, followed by a 10-element softmax 
# output layer.
# 
# Input --> Convolutional(10) --> Maxpool --> Flatten --> 
# Dense (hidden) --> Dense (output).
# 
# This shows how to get going with the MNIST project.  This net is 
# not good enough to achieve the full 99%+ accuracy required on the 
# testing set.
# 
# Some points: Adam optimizer, 80%/20% train/validation split.  Small 
# number of epochs (30), but this is just an upper bound, as we will 
# use early stopping.  The Adam optimizer can be a good choice for
# CNNs, but beware, it can converge to suboptimal solutions.  
# I have found RMSprop to be slightly better in practice.
#
# The size of the first convolutional layer is important, here I'm
# using 3x3, but there are other alternatives....
# 

EPOCHS  = 30      # Training run parameters.  30 epoch run limit.
SPLIT   = 0.2     # 80%/20% train/val split.
SHUFFLE = True    # Shuffle training data before each epoch.
BATCH   = 32      # Minibatch size (note Keras default is 32).
OPT     = 'Adam'  # Adam optimizer.

import numpy as np
import matplotlib.pyplot as plt


# Initialise the random number generators to help reduce variance
# between runs.  Can't eliminate it completely, but this initialisation
# is of some use.


np.random.seed(1)                # Initialise system RNG.

import tensorflow
tensorflow.random.set_seed(2)    # and the seed of the Tensorflow backend.

print(tensorflow.__version__)    # Should be at least 2.0.


# Import the relevant Keras library modules into the IPython notebook.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense          # Fully-connected layer
from tensorflow.keras.layers import Conv2D         # 2-d Convolutional layer
from tensorflow.keras.layers import MaxPooling2D   # 2-d Max-pooling layer
from tensorflow.keras.layers import Flatten        # Converts 2-d layer to 1-d layer
from tensorflow.keras.layers import Activation     # Nonlinearities

from tensorflow.keras.utils import to_categorical


# Load up the MNIST dataset.
#
# This is part of Keras, and will be downloaded automatically from
# the repository on first use.
#

from tensorflow.keras.datasets import mnist


# Extract the training images into "training_inputs" and their
# associated class labels into "training_labels".
#
# Similarly with the testing set, images in in "testing_inputs"
# and labels in "testing_labels".
#
# There are 60000 28 x 28 8-bit greyscale training images and
# 10000 test images.

(training_inputs, training_labels), (testing_inputs, testing_labels) = mnist.load_data()

print(training_inputs.shape, training_inputs.dtype, testing_inputs.shape, testing_inputs.dtype)


# The inputs to the network need to be normalised 'float32'
# values, in a tensor of shape (N,28,28,1).  N is the number of
# images, each one with 28 rows and 28 columns, and one channel.
#
# A greyscale image has one channel (normally implicit), an RGB
# image would have 3. A convolutional net can work with multiple-
# channel input images, but needs the number of channels to be
# explicitly stated, hence the final 1 in the tensor shape.
#
# (This is because the Keras "Conv2D" layer expects 4-tensors
#  as inputs.)
#
# The labels also need to be converted to categorical form.
#

training_images = (training_inputs.astype('float32')/255)[:,:,:,np.newaxis]  # Normalised float32 4-tensor.

categorical_training_outputs = to_categorical(training_labels)

testing_images = (testing_inputs.astype('float32')/255)[:,:,:,np.newaxis]

categorical_testing_outputs = to_categorical(testing_labels)

print(training_images.shape,training_images.dtype)
print(testing_images.shape,testing_images.dtype)
print(categorical_training_outputs.shape, training_labels.shape)
print(categorical_testing_outputs.shape, testing_labels.shape)

plt.figure(figsize=(14,4))
for i in range(20):
    plt.subplot(2,10,i+1)
    plt.imshow(training_images[i,:,:,0],cmap='gray')
    plt.title(str(training_labels[i]))
    plt.axis('off')


# Create a Keras model for a net.
#
# This is a first-guess convolutional model for Optdigits.
#
# It has a single convolutional layer and a single maxpooling
# layer.  The net is then flattened and capped with a hidden
# layer and an output layer.
#
# The convolutional layer uses a 3 x 3 sampling window and has
# 10 slices (3 x 3 x 30).  Edges are padded with copies of the
# input data.
#
# Stride on the convolutional layer is implicitly 1.
#
# The Maxpool layer uses a 2 x 2 sampling window, without
# overlap (i.e., stides of 2 in both x and y).
#
# Finally the network is flattened to 1-d and fed to what is
# essentially a standard hidden+ouput backprop net for
# classification.
#
# These parameters are all fairly ad-hoc...
#

model = Sequential([
            Conv2D(10, kernel_size=3, padding='same', input_shape=training_images.shape[1:]),
            Activation('relu'),
            MaxPooling2D(pool_size=(2,2), strides=(2,2)),
            Flatten(),
            Dense(52),
            Activation('relu'),
            Dense(10),
            Activation('softmax')
        ])


# Review the network, to make sure that Keras has the same ideas that we do.

# In[ ]:


print("The Keras network model")
model.summary()

# Once we have a model and an optimizer, we compile it on the computational
# backend being used by Keras.  When we compile, we specify a loss, or error
# model.
#
# Here, because we have a convolutional network model, Log-Loss is a good
# metric.  In Keras this is called "categorical_crossentropy".  And, if
# using Log-Loss, we really need an adaptive gradient algorithm, here
# I've used RMSprop (Keras name "rmsprop"), which is a good, reliable
# adaptive algoithm.  Adam would also work well.  My metric is accuracy
# on the validation set, but I'm also interested in monitoring training
# accuracy and training/validation loss.
#

model.compile(loss='categorical_crossentropy', optimizer=OPT, metrics=['accuracy'])


# Now we are ready to train.  The routine to do this is "fit".  It tries
# to generate a model that fits the training set.
#
# Parameters to "fit" include, obviously, the training inputs and outputs,
# but, also, the minbatch size used on each training cycle, and the split
# between the training and the validation set.
#
# We limit the number of epoch of training to 30.  But, more importantly,
# we have included an Early Stopping criterion in the run.  This monitors
# the validation loss, and if this starts to increase, halts the training.
# Increasing validation loss (with falling training loss) means that the
# network is starting to overfit, so there is no point in continuing the
# training.  Early stopping is a useful techinique - see the Keras
# documentation for details of the parameters (but the ones used here are
# sensible).
#

from tensorflow.keras.callbacks import EarlyStopping

stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3,
                     verbose=2, mode='auto',
                     restore_best_weights=True)


history = model.fit(training_images, categorical_training_outputs,
                    epochs=EPOCHS,
                    batch_size=BATCH,
                    shuffle=SHUFFLE,
                    validation_split=SPLIT,
                    verbose=2,
                    callbacks=[stop])


# We can now examine the loss (error) per epoch for both the testing
# and the validation data.
#

plt.figure('Training and Validation Losses per epoch', figsize=(8,8))

plt.plot(history.history['loss'],label='training') # Training data error per epoch.

plt.plot(history.history['val_loss'],label='validation') # Validation error per ep.

plt.grid(True)

plt.legend()

plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.show()


# We can also test the performance of the network on the completely separate
# testing set.  We get about 98% accuracy.  Note that this is testing on
# unseen inputs, so is a true measure of performance.
#


print("Performance of network on testing set:")
test_loss,test_acc = model.evaluate(testing_images,categorical_testing_outputs)
print("Accuracy on testing data: {:6.2f}%".format(test_acc*100))
print("Test error (loss):        {:8.4f}".format(test_loss))


# It is also interesting to see the accuracy reported on the training and
# validation data.  The highest accuracy is always reported by the
# training set, validation is worse, and typically better than the accuracy
# reported by testing on the unseen testing set.  Here validation and testing
# accuracies are about the same.
#

print("Performance of network:")
print("Accuracy on training data:   {:6.2f}%".format(history.history['accuracy'][-1]*100))
print("Accuracy on validation data: {:6.2f}%".format(history.history['val_accuracy'][-1]*100))
print("Accuracy on testing data:    {:6.2f}%".format(test_acc*100))


# Suggestions: Try increasing the number of convolutional layers and the
# number of slices in the deeper convolutional layers (with maxpooling
# between the layers for dimensionality reduction).  Consider adding
# (judiciously) a Dropout layer.
#
