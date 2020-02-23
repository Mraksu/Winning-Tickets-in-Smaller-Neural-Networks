'''this file is for retrieving the MNIST dataset in
preprocessed format'''

from tensorflow.keras import datasets
import tensorflow as tf
import numpy as np

def load_MNIST():
    # Load MNIST dataset-
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    print('the shape of the train features is {}'.format(X_train.shape))
    print('the shape of the train label is {}'.format(y_train.shape))

    img_rows,img_cols=X_train.shape[1],X_train.shape[2]
    num_classes=len(list(set(y_train)))


    # In Keras configuration settings, it assumes that the 'channels_last' aproach. We have to change that if it is the case.
    #For 2D data (e.g. image), "channels_last" assumes (rows, cols, channels) while "channels_first" assumes (channels, rows, cols).

    if tf.keras.backend.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    print('new shape of the data after the adding the channel is {}'.format(X_train.shape))

    # Convert datasets to floating point types-
    # Normalize the training and testing datasets-
    X_train = tf.keras.utils.normalize(X_train).astype(np.float32)
    X_test = tf.keras.utils.normalize(X_test).astype(np.float32)

    # convert class vectors/target to binary class matrices or one-hot encoded values-
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # shapes that are going to use for reshaping

    X_train_shape=X_train.shape
    X_test_shape=X_test.shape

    # Reshape training and testing sets-
    X_train = X_train.reshape(X_train_shape[0],X_train_shape[2]*X_train_shape[2] )
    X_test = X_test.reshape(X_test_shape[0],X_test_shape[1]*X_test_shape[2])

    print("\nDimensions of training and testing sets are:")
    print("X_train.shape = {0}, y_train = {1}".format(X_train.shape, y_train.shape))
    print("X_test.shape = {0}, y_test = {1}".format(X_test.shape, y_test.shape))

    return X_train,y_train,X_test,y_test
