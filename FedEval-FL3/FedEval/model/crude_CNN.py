import tensorflow as tf
import os
import numpy as np
import tensorflow_model_optimization as tfmot
from tensorflow.python.keras import regularizers


class crude_CNN():

    def __init__(self, target_shape, **kwargs):
        num_classes = target_shape[-1]

        inputs = tf.keras.layers.Input(shape=(32,32,3))
        # 1st Conv layer 
        x = tf.keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_regularizer=regularizers.l2(0.0001))(inputs)
        x = tf.keras.layers.MaxPool2D(pool_size = (2, 2))(x)
        x = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_regularizer=regularizers.l2(0.0001))(x)
        x = tf.keras.layers.MaxPool2D(pool_size = (2, 2))(x)
        x = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_regularizer=regularizers.l2(0.0001))(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.MaxPool2D(pool_size = (2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation = 'relu', kernel_regularizer=regularizers.l2(0.0003))(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        self.model = model
     
    def compile(self, loss, metrics, optimizer, run_eagerly):
        self.model.compile(loss=loss,
                           metrics=metrics,
                           optimizer=optimizer,
                           run_eagerly=run_eagerly)

    def fit(self, x, y, epochs, batch_size, verbose):
        return self.model.fit(x=x,
                              y=y,
                              epochs=epochs,
                              batch_size=batch_size,
                              verbose=verbose,
                            )


    def evaluate(self, x, y, batch_size, verbose):
        res = self.model.evaluate(x=x,
                                  y=y,
                                  batch_size=batch_size,
                                  verbose=verbose)
        return res

    def summary(self) :
        return self.model.summary()
    
    def compute_output_shape(self, input_shape) :
        pass

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, wgts):
        self.model.set_weights(wgts)

    @property
    def inputs(self) :
        return self.model.inputs

    @property    
    def output_layer(self) :
        return self.model.layers[1].output

    def save_weights(self, path, save_format) :
        self.model.save_weights(os.path.join(path, 'ml_model.h5'), save_format='h5')

    def save_weights(self, path) :
        self.model.save_weights(os.path.join(path, 'ml_model.h5'))

    def load_weights(self, path) :
        self.model.load_weights(os.path.join(path, 'ml_model.h5'))

    def is_crude_CNN() :
        return True
    
    def is_CNN() :
        return True

    @property
    def metrics_names(self) :
        '''
        this value is available after model.fit has been called 
        '''
        return self.model.metrics_names