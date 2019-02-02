import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, Add, Input, Activation
from keras.models import Model
from keras.activations import relu
from keras import backend as K

### Define bottleneck blocks ###

def bottleneck_block_s1(x, end_points, layer_count, expansion_rate=6):
    in_channels = K.int_shape(x)[-1]
    expanded_channels = expansion_rate * in_channels

    ## Expand ##
    endpoint = 'layer_%d_expansion_output'%layer_count
    m = Conv2D(expanded_channels, (1,1), name=endpoint)(x)
    m = BatchNormalization()(m)
    m = Activation('relu')(m)
    end_points[endpoint] = m

    ## DepthWiseConv ##
    endpoint = 'layer_%d_depthwise_output'%layer_count
    m = DepthwiseConv2D(3, padding='same' , name=endpoint)(m)
    m = BatchNormalization()(m)
    m = Activation('relu')(m)
    end_points[endpoint] = m

    ## Project ##
    endpoint = 'layer_%d_projection_output'%layer_count
    m = Conv2D(in_channels, (1,1), name=endpoint)(m)
    m = BatchNormalization()(m)
    end_points[endpoint] = m

    return Add()([m,x])

def bottleneck_block_s2(x, out_channels, end_points, layer_count, expansion_rate=6, stride=2):
    in_channels = K.int_shape(x)[-1]
    expanded_channels = expansion_rate * in_channels
    #print('in_channels: {} | expanded_channels: {}'.format(in_channels, expanded_channels))

    ## Expand ##
    endpoint = 'layer_%d_expansion_output'%layer_count
    m = Conv2D(expanded_channels, (1,1), name=endpoint)(x)
    m = BatchNormalization()(m)
    m = Activation('relu')(m)
    end_points[endpoint] = m

    ## DepthWiseConv ##
    endpoint = 'layer_%d_depthwise_output'%layer_count
    m = DepthwiseConv2D(3, padding='same', strides=stride, name=endpoint)(m)
    m = BatchNormalization()(m)
    m = Activation('relu')(m)
    end_points[endpoint] = m

    ## Project and change original channel count ##
    endpoint = 'layer_%d_projection_output'%layer_count
    m = Conv2D(out_channels, (1,1), name=endpoint)(m)
    m = BatchNormalization()(m)
    end_points[endpoint] = m

    return m

### Define a sequence of Bottleneck blocks ###
def bottleneck_sequence(x, end_points, layer_count, out_channels='same', stride=1, expansion_rate=6, count=1):
    block_count = count

    m = x
    if out_channels != 'same' :
        m = bottleneck_block_s2(x, out_channels, end_points, layer_count, expansion_rate=expansion_rate, stride=stride)
        block_count -= 1
        layer_count +=1

    while block_count > 0 :
        m = bottleneck_block_s1(m, end_points, layer_count, expansion_rate=expansion_rate)
        block_count -= 1
        layer_count += 1

    return m

### Define the MobileNetV2 Feature Extractor model ###

def MobileNetV2_FE(x, output_stride=16):
    '''
    MobileNetV2 Feature Extractor
    Returns the output of the last 'Bottleneck' sequence
    Additionally returns early image-level features if return_IF is True

    Supports only output_stride=16 or 8
    '''
    end_points = {}
    layer_count = 1

    endpoint = 'layer_%d' % layer_count
    m = Conv2D(32, (3,3), padding='same', strides=2, name=endpoint)(x)
    m = BatchNormalization()(m)
    m = Activation('relu')(m)
    end_points[endpoint] = m
    layer_count += 1

    m = bottleneck_sequence(m, end_points, layer_count, out_channels=16, expansion_rate=1)
    layer_count += 1

    m = bottleneck_sequence(m, end_points, layer_count, out_channels=24, stride=2, count=2)
    layer_count += 2

    m = bottleneck_sequence(m, end_points, layer_count, out_channels=32, stride=2, count=3)
    layer_count += 3

    if output_stride == 8:
        m = bottleneck_sequence(m, end_points, layer_count, out_channels=64, stride=1, count=4)
    else:
        m = bottleneck_sequence(m, end_points, layer_count, out_channels=64, stride=2, count=4)
    layer_count += 4


    m = bottleneck_sequence(m, end_points, layer_count, out_channels=96, stride=1, count=3)
    layer_count += 3

    ## Making Stride=1 for the next block since we want a maximum of 16
    ## as output_stride
    m = bottleneck_sequence(m, end_points, layer_count, out_channels=160, stride=1, count=3)
    layer_count += 3

    m = bottleneck_sequence(m, end_points, layer_count, out_channels=320, stride=1, count=1)

    #m = Conv2D(1280, (1,1))(m)
    #m = BatchNormalization()(m)

    return m, end_points

if __name__ == '__main__':
    print('Creating the model for testing with input shape (16,16,3)...')
    input_image = Input(shape=(16,16,3))
    mbnet_features, end_points = MobileNetV2_FE(input_image)
    model = Model(input_image, mbnet_features)
    n = np.random.rand(16,16,3)
    n = np.expand_dims(n, 0)
    t = model.predict(n)

    print('model.predict() output shape: {} \n Output:\n {}'.format(t.shape, t))
