"""
DOCSTRING
"""
import tensorflow

WEIGHTS_INIT_STDEV = 0.1

def _conv_init_vars(net, out_channels, filter_size, transpose=False):
    """
    DOCSTRING
    """
    _, rows, cols, in_channels = [i.value for i in net.get_shape()]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]
    weights_init = tensorflow.Variable(tensorflow.truncated_normal(
        weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tensorflow.float32)
    return weights_init

def _conv_layer(net, num_filters, filter_size, strides, relu=True):
    """
    DOCSTRING
    """
    weights_init = _conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    net = tensorflow.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
    net = _instance_norm(net)
    if relu:
        net = tensorflow.nn.relu(net)
    return net

def _conv_tranpose_layer(net, num_filters, filter_size, strides):
    """
    DOCSTRING
    """
    weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True)
    batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
    new_rows, new_cols = int(rows * strides), int(cols * strides)
    #new_shape = tensorflow.stack(
    #    [tensorflow.shape(net)[0], new_rows, new_cols, num_filters])
    new_shape = [batch_size, new_rows, new_cols, num_filters]
    tf_shape = tensorflow.stack(new_shape)
    strides_shape = [1,strides,strides,1]
    net = tensorflow.nn.conv2d_transpose(
        net, weights_init, tf_shape, strides_shape, padding='SAME')
    net = _instance_norm(net)
    return tensorflow.nn.relu(net)

def _instance_norm(net, train=True):
    """
    DOCSTRING
    """
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tensorflow.nn.moments(net, [1,2], keep_dims=True)
    shift = tensorflow.Variable(tensorflow.zeros(var_shape))
    scale = tensorflow.Variable(tensorflow.ones(var_shape))
    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
    return scale * normalized + shift

def _residual_block(net, filter_size=3):
    """
    DOCSTRING
    """
    tmp = _conv_layer(net, 128, filter_size, 1)
    return net + _conv_layer(tmp, 128, filter_size, 1, relu=False)

def net(image):
    """
    DOCSTRING
    """
    conv1 = _conv_layer(image, 32, 9, 1)
    conv2 = _conv_layer(conv1, 64, 3, 2)
    conv3 = _conv_layer(conv2, 128, 3, 2)
    resid1 = _residual_block(conv3, 3)
    resid2 = _residual_block(resid1, 3)
    resid3 = _residual_block(resid2, 3)
    resid4 = _residual_block(resid3, 3)
    resid5 = _residual_block(resid4, 3)
    conv_t1 = _conv_tranpose_layer(resid5, 64, 3, 2)
    conv_t2 = _conv_tranpose_layer(conv_t1, 32, 3, 2)
    conv_t3 = _conv_layer(conv_t2, 3, 9, 1, relu=False)
    preds = tensorflow.nn.tanh(conv_t3) * 150 + 255./2
    return preds
