from tensorflow.keras.layers import *

# Custom Conv/Deconv + Batch Norm Layers from DHB echo model
# --------------------------------------------------- Custom Conv + Batch Norm Layer ---------------------------------------
class Conv(Layer):

    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', activation=None, batch_normalisation=False, name=None):
        super(Conv, self).__init__()

        self.activation = activation
        self.batch_normalisation = batch_normalisation

        self.conv = Conv2D(filters, kernel_size, activation=None, strides=strides, padding=padding, name=name, use_bias=(not batch_normalisation))

        if self.batch_normalisation:
            self.bn = BatchNormalization()

    def call(self, inputs, training=False):

        # apply conv then batch norm then apply the activation
        h = inputs
        h = self.conv(h)
        if self.batch_normalisation:
            h = self.bn(h, training=training)
        h = Activation(self.activation)(h)
        return h
# --------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------- Custom Deconv + Batch Norm Layer -------------------------------------
class DeConv(Layer):

    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', activation=None, batch_normalisation=False, output_padding=None, name=None):
        super(DeConv, self).__init__()

        self.activation = activation
        self.batch_normalisation = batch_normalisation

        self.conv = Conv2DTranspose(filters, kernel_size, strides=strides, activation=None, padding=padding, output_padding=output_padding, use_bias=(not batch_normalisation), name=name)

        if self.batch_normalisation:
            self.bn = BatchNormalization()

    def call(self, inputs, training=False):

        # apply deconv then batch norm then apply the activation
        h = inputs
        h = self.conv(h)
        if self.batch_normalisation:
            h = self.bn(h, training=training)
        h = Activation(self.activation)(h)
        return h

# --------------------------------------------------------------------------------------------------------------------------