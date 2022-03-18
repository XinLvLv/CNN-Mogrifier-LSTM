# Implementation of Inception-v4 architecture
# Author: Shobhit Lamba
# e-mail: slamba4@uic.edu

# Importing the libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import os
import warnings
warnings.filterwarnings("ignore")
from keras.models import Model
from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
import imagenet_utils
from imagenet_utils import _obtain_input_shape
from imagenet_utils import decode_predictions
from keras import backend as K

BASE_WEIGHT_URL = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.7/'

def conv_block(x, nb_filter, nb_row, nb_col, padding="same", strides=(1, 1), use_bias=False):
    '''Defining a Convolution block that will be used throughout the network.'''

    x = Conv2D(nb_filter, (nb_row, nb_col), strides=strides, padding=padding, use_bias=use_bias)(x)
    x = BatchNormalization(axis=-1, momentum=0.9997, scale=False)(x)
    x = Activation("relu")(x)

    return x

def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size as in `Conv2D`.
        strides: strides in `Conv2D`.
        padding: padding mode in `Conv2D`.
        activation: activation in `Conv2D`.
        use_bias: whether to use a bias in `Conv2D`.
        name: name of the ops; will become `name + '_ac'` for the activation
            and `name + '_bn'` for the batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = Conv2D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               name=name)(x)
    if not use_bias:
        bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
        bn_name = None if name is None else name + '_bn'
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = Activation(activation, name=ac_name)(x)
    return x


def stem(input):
    '''The stem of the pure Inception-v4 and Inception-ResNet-v2 networks. This is input part of those networks.'''

    # Input shape is 299 * 299 * 3 (Tensorflow dimension ordering)
    x = conv_block(input, 32, 3, 3, strides=(2, 2), padding="same")  # 149 * 149 * 32
    x = conv_block(x, 32, 3, 3, padding="same")  # 147 * 147 * 32
    x = conv_block(x, 64, 3, 3)  # 147 * 147 * 64
    y = x

    x1 = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x2 = conv_block(x, 96, 3, 3, strides=(2, 2), padding="same")

    x = concatenate([x1, x2], axis=-1)  # 73 * 73 * 160

    x1 = conv_block(x, 64, 1, 1)
    x1 = conv_block(x1, 96, 3, 3, padding="same")

    x2 = conv_block(x, 64, 1, 1)
    x2 = conv_block(x2, 64, 1, 7)
    x2 = conv_block(x2, 64, 7, 1)
    x2 = conv_block(x2, 96, 3, 3, padding="same")

    x = concatenate([x1, x2], axis=-1)  # 71 * 71 * 192

    x1 = conv_block(x, 192, 3, 3, strides=(2, 2), padding="same")

    x2 = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    x = concatenate([x1, x2], axis=-1)  # 35 * 35 * 384

    return x,y


def inception_A(input):
    '''Architecture of Inception_A block which is a 35 * 35 grid module.'''

    a1 = AveragePooling2D((3, 3), strides=(1, 1), padding="same")(input)
    a1 = conv_block(a1, 96, 1, 1)

    a2 = conv_block(input, 96, 1, 1)

    a3 = conv_block(input, 64, 1, 1)
    a3 = conv_block(a3, 96, 3, 3)

    a4 = conv_block(input, 64, 1, 1)
    a4 = conv_block(a4, 96, 3, 3)
    a4 = conv_block(a4, 96, 3, 3)

    merged = concatenate([a1, a2, a3, a4], axis=-1)

    return merged


def inception_B(input):
    '''Architecture of Inception_B block which is a 17 * 17 grid module.'''

    b1 = AveragePooling2D((3, 3), strides=(1, 1), padding="same")(input)
    b1 = conv_block(b1, 128, 1, 1)

    b2 = conv_block(input, 384, 1, 1)

    b3 = conv_block(input, 192, 1, 1)
    b3 = conv_block(b3, 224, 1, 7)
    b3 = conv_block(b3, 256, 7, 1)

    b4 = conv_block(input, 192, 1, 1)
    b4 = conv_block(b4, 192, 7, 1)
    b4 = conv_block(b4, 224, 1, 7)
    b4 = conv_block(b4, 224, 7, 1)
    b4 = conv_block(b4, 256, 1, 7)

    merged = concatenate([b1, b2, b3, b4], axis=-1)

    return merged


def inception_C(input):
    '''Architecture of Inception_C block which is a 8 * 8 grid module.'''

    c1 = AveragePooling2D((3, 3), strides=(1, 1), padding="same")(input)
    c1 = conv_block(c1, 256, 1, 1)

    c2 = conv_block(input, 256, 1, 1)

    c3 = conv_block(input, 384, 1, 1)
    c31 = conv_block(c2, 256, 1, 3)
    c32 = conv_block(c2, 256, 3, 1)
    c3 = concatenate([c31, c32], axis=-1)

    c4 = conv_block(input, 384, 1, 1)
    c4 = conv_block(c3, 448, 3, 1)
    c4 = conv_block(c3, 512, 1, 3)
    c41 = conv_block(c3, 256, 1, 3)
    c42 = conv_block(c3, 256, 3, 1)
    c4 = concatenate([c41, c42], axis=-1)

    merged = concatenate([c1, c2, c3, c4], axis=-1)

    return merged


def reduction_A(input, k=192, l=224, m=256, n=384):
    '''Architecture of a 35 * 35 to 17 * 17 Reduction_A block.'''

    ra1 = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(input)

    ra2 = conv_block(input, n, 3, 3, strides=(2, 2), padding="same")

    ra3 = conv_block(input, k, 1, 1)
    ra3 = conv_block(ra3, l, 3, 3)
    ra3 = conv_block(ra3, m, 3, 3, strides=(2, 2), padding="same")

    merged = concatenate([ra1, ra2, ra3], axis=-1)

    return merged


def reduction_B(input):
    '''Architecture of a 17 * 17 to 8 * 8 Reduction_B block.'''

    rb1 = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(input)

    rb2 = conv_block(input, 192, 1, 1)
    rb2 = conv_block(rb2, 192, 3, 3, strides=(2, 2), padding="same")

    rb3 = conv_block(input, 256, 1, 1)
    rb3 = conv_block(rb3, 256, 1, 7)
    rb3 = conv_block(rb3, 320, 7, 1)
    rb3 = conv_block(rb3, 320, 3, 3, strides=(2, 2), padding="same")

    merged = concatenate([rb1, rb2, rb3], axis=-1)

    return merged


def inception_v4(include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 classes=1000):
    '''Creates the Inception_v4 network.'''
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape  新增
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=299,
        min_size=139,
        data_format=K.image_data_format(),
        require_flatten=False,
        weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # init = Input((299, 299, 3)) # Channels last, as using Tensorflow backend with Tensorflow image dimension ordering

    # Input shape is 299 * 299 * 3
    #  # Output: 35 * 35 * 384
    x,y = stem(img_input)

    # 4 x Inception A
    for i in range(4):
        x = inception_A(x)
        # Output: 35 * 35 * 384

    # Reduction A
    x = reduction_A(x, k=192, l=224, m=256, n=384)  # Output: 17 * 17 * 1024

    # 7 x Inception B
    for i in range(7):
        x = inception_B(x)
        # Output: 17 * 17 * 1024

    # Reduction B
    x = reduction_B(x)  # Output: 8 * 8 * 1536

    # 3 x Inception C
    for i in range(3):
        x = inception_C(x)
        # Output: 8 * 8 * 1536

    y = conv_block(y, 1536, 1,1)
    x = x+y

    # Average Pooling
    #x = AveragePooling2D((8, 8))(x)  # Output: 1536

    # Dropout
    #x = Dropout(0.2)(x)  # Keep dropout 0.2 as mentioned in the paper
    #x = Flatten()(x)  # Output: 1536

    # Output layer
    #output = Dense(units=classes, activation="softmax")(x)  # Output: 1000

    # Final convolution block: 8 x 8 x 1536
    x = conv2d_bn(x, 1536, 1, name='conv_7b')

    if include_top:
        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        output = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            output = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            output = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`
    if input_tensor is not None:
        init = get_source_inputs(input_tensor)
    else:
        init = img_input

    model = Model(init, output, name="Inception-v4")

    # Load weights
    if weights == 'imagenet':
        if K.image_data_format() == 'channels_first':
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
        if include_top:
            fname = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5'
            weights_path = get_file(fname,
                                    BASE_WEIGHT_URL + fname,
                                    cache_subdir='models',
                                    file_hash='e693bd0210a403b3192acc6073ad2e96')
        else:
            fname = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
            weights_path = get_file(fname,
                                    BASE_WEIGHT_URL + fname,
                                    cache_subdir='models',
                                    file_hash='d19885ff4a710c122648d3b5c3b684e4')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model
