"""
Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.
"""
import numpy
import scipy.io
import tensorflow

MEAN_PIXEL = numpy.array([ 123.68 ,  116.779,  103.939])

def _conv_layer(input, weights, bias):
    """
    DOCSTRING
    """
    conv = tensorflow.nn.conv2d(
        input, tensorflow.constant(weights), strides=(1, 1, 1, 1), padding='SAME')
    return tensorflow.nn.bias_add(conv, bias)

def _pool_layer(input):
    """
    DOCSTRING
    """
    return tensorflow.nn.max_pool(
        input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

def net(data_path, input_image):
    """
    DOCSTRING
    """
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4')
    data = scipy.io.loadmat(data_path)
    mean = data['normalization'][0][0][0]
    mean_pixel = numpy.mean(mean, axis=(0, 1))
    weights = data['layers'][0]
    net = {}
    current = input_image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = numpy.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias)
        elif kind == 'relu':
            current = tensorflow.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current)
        net[name] = current
    assert len(net) == len(layers)
    return net

def preprocess(image):
    return image - MEAN_PIXEL

def unprocess(image):
    return image + MEAN_PIXEL
