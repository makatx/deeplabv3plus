
# coding: utf-8

# In[42]:


import numpy as np
from keras.layers import Conv2D, SeparableConv2D, Input, Activation, BatchNormalization, DepthwiseConv2D, Dropout, Add, Lambda
from keras import backend as K
from mobilenetv2FE import MobileNetV2_FE
from keras.models import Model
import tensorflow as tf

## Define some utility functions

def split_separable_conv2d(x, filters, kernel_size, strides=(1, 1), padding='same', dilation_rate=1, activation=None):
    '''
    Splits the separable conv into depthwise and pointwise and then applies activation between them as well
    '''
    #BN = BatchNormalization(momentum=0.9997, epsilon=1e-5)

    m = DepthwiseConv2D(kernel_size, strides, padding=padding, dilation_rate=dilation_rate)(x)
    m = BatchNormalization(momentum=0.9997, epsilon=1e-5)(m)
    m = Activation('relu')(m)

    m = Conv2D(filters, kernel_size=1)(m)
    #m = BN(m)
    m = BatchNormalization(momentum=0.9997, epsilon=1e-5)(m)
    m = Activation('relu')(m)

    return m

def ReduceMeanLayer(features, axis=[1,2], keepdims=True):
    func = lambda features, axis, keepdims: tf.reduce_mean(features, axis=axis, keepdims=keepdims)
    return Lambda(func, arguments={'axis':axis, 'keepdims':keepdims})(features)


def ResizeImageLayer(images, hw, align_corners=True):
    func = lambda images, hw, align_corners: tf.image.resize_bilinear(images, hw, align_corners)
    return Lambda(func, arguments={'hw':hw, 'align_corners':align_corners})(images)

def ConcatenateLayer(tensors, axis=3):
    func = lambda tensors, axis: K.concatenate(tensors, axis)
    return Lambda(func, arguments={'axis':axis})(tensors)


def extract_features(x, atrous_rates=None, output_stride=16, include_image_level_features=True):

    ## Get features from mobilenet
    features, endpoints = MobileNetV2_FE(x, output_stride)

    branch_logits = []
    depth = 256
    BN = BatchNormalization(momentum=0.9997, epsilon=1e-5)

    if include_image_level_features:
        ## Perform Global Avg Pool and resize back to the features' image dims
        image_features = ReduceMeanLayer(features, axis=[1,2], keepdims=True)

        image_features = Conv2D(depth, 1)(image_features)
        image_features = BatchNormalization(momentum=0.9997, epsilon=1e-5)(image_features)
        image_features = Activation('relu')(image_features)

        resize_height = K.shape(features)[1]
        resize_width = K.shape(features)[2]
        image_features = ResizeImageLayer(image_features, [resize_height, resize_width], align_corners=True)
        #image_features.set_shape([None, resize_height, resize_width, depth])

        branch_logits.append(image_features)

    FE_logits = Conv2D(depth, 1)(features)
    FE_logits = BatchNormalization(momentum=0.9997, epsilon=1e-5)(FE_logits)
    FE_logits = Activation('relu')(FE_logits)
    branch_logits.append(FE_logits)

    if atrous_rates:
        for rate in atrous_rates:
            aspp_features = split_separable_conv2d(features, depth, 3, dilation_rate=rate)
            branch_logits.append(aspp_features)
    
    if len(branch_logits) == 1:
        concat_logits = branch_logits[0]
    else:    
        concat_logits = ConcatenateLayer(branch_logits, axis=3)
    concat_logits = Conv2D(depth, 1 )(concat_logits)
    concat_logits = BatchNormalization(momentum=0.9997, epsilon=1e-5)(concat_logits)
    concat_logits = Activation('relu')(concat_logits)
    concat_logits = Dropout(0.1)(concat_logits)

    return concat_logits, endpoints

def refine_by_decoder(features, endpoints, decoder_height, decoder_width, atrous_rates=None, use_separable_conv=True):
    '''
    Gets and combines (concat) low-level features (from mobilenet layer 4) and features from extract_features() function
    after sampling them to decoder_height/width; then performs split_separable_conv operation twice
    '''
    BN = BatchNormalization(momentum=0.9997, epsilon=1e-5)
    decoder_depth = 256

    ## Conv to get 48 channels from mobilenet layer 4
    low_level_features = endpoints['layer_4_depthwise_output']
    low_level_features = Conv2D(48, 1)(low_level_features)
    low_level_features = BatchNormalization(momentum=0.9997, epsilon=1e-5)(low_level_features)
    low_level_features = Activation('relu')(low_level_features)

    decoder_features_list = [features]
    decoder_features_list.append(low_level_features)

    ## Upsample low_level_features and features to provided dimensions
    for i, feature in enumerate(decoder_features_list):
        decoder_features_list[i] = ResizeImageLayer(feature, [decoder_height, decoder_width], align_corners=True)
        h = (None if isinstance(decoder_height, tf.Tensor)
             else decoder_height)
        w = (None if isinstance(decoder_width, tf.Tensor)
             else decoder_width)
        decoder_features_list[i].set_shape([None, h, w, None])

    decoder_features = ConcatenateLayer(decoder_features_list, axis=3)

    if use_separable_conv:
        decoder_features = split_separable_conv2d(decoder_features, decoder_depth, 3, padding='same')
        decoder_features = split_separable_conv2d(decoder_features, decoder_depth, 3, padding='same')
    else:
        decoder_features = Conv2D(decoder_depth, 3, padding='same')(decoder_features)
        decoder_features = BatchNormalization(momentum=0.9997, epsilon=1e-5)(decoder_features)
        decoder_features = Activation('relu')(decoder_features)

        decoder_features = Conv2D(decoder_depth, 3, padding='same')(decoder_features)
        decoder_features = BatchNormalization(momentum=0.9997, epsilon=1e-5)(decoder_features)
        decoder_features = Activation('relu')(decoder_features)

    return decoder_features

def branch_and_merge(features, num_classes, atrous_rates=None):
    '''
    Applies atrous_rates in parallel to features and then sum-merges them
    If atrous_rates is None then applies just one Conv2D
    Out_channels is equal to num_classes

    No Activations or Normalizations are applied
    '''

    if atrous_rates == None:
        atrous_rates = [1]

    branch_logits = []

    for rate in atrous_rates:
        branch_logits.append(Conv2D(num_classes, 1)(features))

    print('len(atrous_rates): {}'.format(len(atrous_rates)))
    if len(atrous_rates) == 1:
        print('got None rates. Branch logits: {}'.format(branch_logits))
        return branch_logits[0]
    else:
        return Add()(branch_logits)


def build_deeplab(images,
                  num_classes,
                  output_stride=16,
                  decoder_stride=4,
                  atrous_rates=[3,6,9],
                  use_separable_conv=True, 
                  include_image_level_features=True):

    features, endpoints = extract_features(images, 
                                           atrous_rates=atrous_rates, 
                                           output_stride=output_stride, 
                                           include_image_level_features=include_image_level_features)

    image_height = tf.shape(images)[1]
    image_width = tf.shape(images)[2]
    decoder_height = image_height // decoder_stride
    decoder_width = image_width // decoder_stride

    decoder_features = refine_by_decoder(features, endpoints, decoder_height, decoder_width, atrous_rates, use_separable_conv)

    logits = branch_and_merge(decoder_features, num_classes, atrous_rates)

    ## Upsample to image dimensions
    logits = ResizeImageLayer(logits, [image_height, image_width], align_corners=True)

    return logits


if __name__ == '__main__':
    
    batch_size = 512
    print('\n\nRunning test on model. Creating model with input shape ({},512,512,3)...\n\n'.format(batch_size))

    inp = Input(shape=(512,512,3,))
    logits = build_deeplab(inp, 1)
    model = Model(inp, logits)
    test_inp = np.random.rand(batch_size,512,512,3)
    op = model.predict(test_inp)

    print('Deeplabv3+ model built and run, resulting output shape: {}'.format(op.shape))
