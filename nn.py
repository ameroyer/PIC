from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys
import utils
is_python3 = (sys.version_info[0] == 3)
if is_python3:
    from pickle import load as pickle_load
else:
    import cPickle
    def pickle_load(file, **kwargs):
        cPickle.load(file)


"""
General functions
"""

def get_variable(name, shape=None, dtype=None, initializer=None, trainable=True, regularizer=None):
    """Creates a CPU variable with the given arguments."""
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape=shape, dtype=dtype,
                              initializer=initializer, regularizer=regularizer, trainable=trainable,
                              collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])

        if ('ema' in tf.GLOBAL) and (var in tf.trainable_variables()):
            var = tf.GLOBAL['ema'].average(var)

        return var

def concat_elu(x):
    """Concat elu activation (ELU(x @ -x) concatenated on the last axis)."""
    axis = len(x.get_shape()) - 1
    return tf.nn.elu(tf.concat([x, -x], axis))

def elu(x):
    """elu activation"""
    return tf.nn.elu(x)

def conv(inp, name, filter_size, out_channels, stride=1,
         padding='SAME', nonlinearity=None, init_scale=1.0, dilation=None):
    """Convolutional layer.
    If tf.GLOBAL['init'] is true, this creates the layers paramenters (g, b, W) : L(x) = g|W| (*) x + b

    Args:
      x: input tensor
      name (str): variable scope name
      filter_size (int pair): filter size
      out_channels (int): number of output channels
      strid (int): horizontal and vertical stride
      padding (str): padding mode
      nonlinearity (func): activation function
      init_scale: initial scale for the weights and bias variables
      dilation: optional dilation rate
    """
    with tf.variable_scope(name):
        strides = [1, stride, stride, 1]
        in_channels = inp.get_shape().as_list()[3]

        if tf.GLOBAL['init']:
            V = get_variable('V', shape=tuple(filter_size) + (in_channels, out_channels), dtype=tf.float32,
                             initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0, 1, 2])
            if dilation is None:
                out = tf.nn.conv2d(inp, V_norm, strides, padding)
            else:
                assert(stride == 1)
                out = tf.nn.atrous_conv2d(inp, V_norm, dilation, padding)
            m_init, v_init = tf.nn.moments(out, [0, 1, 2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-8)
            g = get_variable('g', shape=None, dtype=tf.float32, initializer=scale_init, trainable=True, regularizer=tf.contrib.layers.l2_regularizer(tf.GLOBAL['reg']))
            b = get_variable('b', shape=None, dtype=tf.float32, initializer=-m_init * scale_init, trainable=True, regularizer=tf.contrib.layers.l2_regularizer(tf.GLOBAL['reg']))
            out = tf.reshape(scale_init, [1, 1, 1, out_channels]) * (out - tf.reshape(m_init, [1, 1, 1, out_channels]))
            if nonlinearity is not None:
                out = nonlinearity(out)

        else:
            V, g, b = get_variable('V'), get_variable('g'), get_variable('b')
            tf.assert_variables_initialized([V, g, b])
            W = g[None, None, None] * tf.nn.l2_normalize(V, [0, 1, 2])
            if dilation is None:
                out = tf.nn.conv2d(inp, W, strides, padding) + b[None, None, None]
            else:
                assert(stride == 1)
                out = tf.nn.atrous_conv2d(inp, W, dilation, padding) + b[None, None, None]
            if nonlinearity is not None:
                out = nonlinearity(out)

        return out

def deconv(inp, name, filter_size, out_channels, stride=1,
           padding='SAME', nonlinearity=None, init_scale=1.0):
    """ Deconvolution layer. See `conv`"""
    with tf.variable_scope(name):
        strides = [1, stride, stride, 1]
        [N, H, W, in_channels] = inp.get_shape().as_list()
        if padding == 'SAME':
            target_shape = [N, H * stride, W * stride, out_channels]
        else:
            target_shape = [N, H * stride + filter_size[0] - 1, W * stride + filter_size[1] - 1, out_channels]
        target_shape = tf.constant(target_shape, dtype=tf.int32)

        if tf.GLOBAL['init']:
            V = get_variable('V', shape=filter_size + (out_channels, in_channels), dtype=tf.float32,
                             initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0, 1, 3])
            out = tf.nn.conv2d_transpose(inp, V_norm, target_shape, strides, padding)
            m_init, v_init = tf.nn.moments(out, [0, 1, 2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-8)
            g = get_variable('g', shape=None, dtype=tf.float32, initializer=scale_init, trainable=True, regularizer=tf.contrib.layers.l2_regularizer(tf.GLOBAL['reg']))
            b = get_variable('b', shape=None, dtype=tf.float32, initializer=-m_init * scale_init, trainable=True, regularizer=tf.contrib.layers.l2_regularizer(tf.GLOBAL['reg']))
            out = tf.reshape(scale_init, [1, 1, 1, out_channels]) * (out - tf.reshape(m_init, [1, 1, 1, out_channels]))
            if nonlinearity is not None:
                out = nonlinearity(out)

        else:
            V, g, b = get_variable('V'), get_variable('g'), get_variable('b')
            tf.assert_variables_initialized([V, g, b])
            W = g[None, None, :, None] * tf.nn.l2_normalize(V, [0, 1, 3])
            out = tf.nn.conv2d_transpose(inp, W, target_shape, strides, padding) + b[None, None, None]
            if nonlinearity is not None:
                out = nonlinearity(out)

        return out

"""
Pixel-CNN functions
"""

def gated_resnet(x, name, nonlinearity=concat_elu, conv=conv, a=None, dilation=None):
    """Gated Resnet."""
    with tf.variable_scope(name):
        num_filters = int(x.get_shape()[-1])
        c1 = conv(nonlinearity(x), "conv1", (3, 3), num_filters, dilation=dilation)

        if a is not None:
            c1 += conv(nonlinearity(a), "conv_aux", (1, 1), num_filters)

        c1 = nonlinearity(c1)
        if tf.GLOBAL['dropout'] > 0:
            c1 = tf.nn.dropout(c1, keep_prob=1.0 - tf.GLOBAL['dropout'])

        c2 = conv(c1, "conv2", (3, 3), num_filters * 2, init_scale=0.1)
        a, b = tf.split(c2, 2, 3)
        c3 = a * tf.nn.sigmoid(b)
        return x + c3

def down_shift(x):
    """Down shift the input by offset 1."""
    xs = int_shape(x)
    return tf.concat([tf.zeros([xs[0], 1, xs[2], xs[3]]), x[:, :xs[1] - 1, :, :]], 1)

def right_shift(x):
    """Right shift the input by offset 1."""
    xs = int_shape(x)
    return tf.concat([tf.zeros([xs[0], xs[1], 1, xs[3]]), x[:, :, :xs[2] - 1, :]], 2)

def down_shifted_conv2d(x, name, filter_size, out_channels, stride=1, nonlinearity=None, init_scale=1.0, dilation=None):
    """Down shifted convolution."""
    x = tf.pad(x, [[0, 0], [filter_size[0] - 1, 0], [int((filter_size[1] - 1) / 2), int((filter_size[1] - 1) / 2)], [0, 0]])
    return conv(x, name, filter_size, out_channels, stride=stride, padding='VALID', nonlinearity=nonlinearity, init_scale=init_scale)

def down_right_shifted_conv2d(x, name, filter_size, out_channels, stride=1, nonlinearity=None, init_scale=1.0, dilation=None):
    """Down-right shifted convolution."""
    x = tf.pad(x, [[0, 0], [filter_size[0] - 1, 0], [filter_size[1] - 1, 0], [0, 0]])
    return conv(x, name, filter_size, out_channels, stride=stride, padding='VALID', nonlinearity=nonlinearity, init_scale=init_scale)

def down_shifted_deconv2d(x, name, filter_size, out_channels, stride=1, nonlinearity=None, init_scale=1.0):
    """Down shifted deconvolution."""
    x = deconv(x, name, filter_size, out_channels, padding='VALID', stride=stride, nonlinearity=nonlinearity, init_scale=init_scale)
    xs = int_shape(x)
    return x[:, :(xs[1] - filter_size[0] + 1), int((filter_size[1] - 1) / 2):(xs[2] - int((filter_size[1] - 1) / 2)), :]

def down_right_shifted_deconv2d(x, name, filter_size, out_channels, stride=1, nonlinearity=None, init_scale=1.0):
    """Down-right shifted convolution."""
    x = deconv(x, name, filter_size, out_channels, padding='VALID', stride=stride, nonlinearity=nonlinearity, init_scale=init_scale)
    xs = int_shape(x)
    return x[:, :(xs[1] - filter_size[0] + 1):, :(xs[2] - filter_size[1] + 1), :]


"""
Log-likelihood computation (see OpenAI code)
"""

def int_shape(x):
    """Returns the shape of the input tensors as a list of ints."""
    return list(map(int, x.get_shape()))

def log_sum_exp(x):
    """Numerically stable log_sum_exp implementation that prevents overflow."""
    axis = len(x.get_shape())-1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keep_dims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x-m2), axis))

def log_prob_from_logits(x):
    """Numerically stable log_softmax implementation that prevents overflow."""
    axis = len(x.get_shape())-1
    m = tf.reduce_max(x, axis, keep_dims=True)
    return x - m - tf.log(tf.reduce_sum(tf.exp(x-m), axis, keep_dims=True))

def colorization_loss(x, l, nr_mix=10, colorspace="RGB", sum_all=True):
    """ Main loss function. Per pixel normalized (but not per batch yet)"""
    xs = int_shape(x)
    return discretized_mix_logistic_loss(x, l, nr_mix, colorspace, sum_all) / (np.log(2.) * xs[1] * xs[2] * (3. if colorspace == "RGB" else 2))

def discretized_mix_logistic_loss(x, l, nr_mix=10, colorspace="RGB", sum_all=True):
    """Log-likelihood for mixture of discretized logistics.

    Args:
      x: input image batch, normalized between [-1, 1]
      l: logistic mixture parametrization
      colorspace (str): colorspace for the input image

    Returns:
      loss: log-likelihood
    """
    xs = int_shape(x)  # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = int_shape(l)    # predicted distribution, e.g. (B,32,32,100)

    if colorspace == "RGB":
        """ RGB mode
        p(R) = Log(mu_R, s_R)
        p(G) = Log(mu_G + c0 * x_R, s_G)
        p(B) = Log(mu_B + c1 * x_R + c2 * x_G, s_B)
        """
        # reshape output: Get means, scales and coefficients
        logit_probs = l[:, :, :, :nr_mix]
        l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
        means = l[:, :, :, :, :nr_mix]
        log_scales = tf.maximum(l[:, :, :, :, nr_mix:2*nr_mix], -7.)
        coeffs = tf.nn.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
        x = tf.reshape(x, xs + [1]) + tf.zeros(xs + [nr_mix])
        # Write Channels dependecy
        m1 = tf.reshape(means[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix])
        m2 = tf.reshape(means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix])
        m3 = tf.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :], [xs[0], xs[1], xs[2], 1, nr_mix])
        means = tf.concat([m1, m2, m3], 3)

    elif colorspace == "lab":
        """  luminance - chrominances colorspace (L,C1,C2)
        chrominance components assumed independent from the luminance
        p(C1) = Log(mu_1 + c0 * L, s_1)
        p(C2) = Log(mu_2 + c1 * L + c2 * C1, s_2)
        """
        L = tf.expand_dims(x[:, :, :, 0], -1)
        x = x[:, :, :, 1:]
        # Extract parameters
        xs = int_shape(x)
        ls = int_shape(l)
        x = tf.reshape(x, xs + [1]) + tf.zeros(xs + [nr_mix])  # (B,32,32,2,K)
        logit_probs = l[:, :, :, :nr_mix]                                                   # Probs (K)
        coeffs = tf.nn.tanh(tf.reshape(l[:, :, :, nr_mix:4*nr_mix], xs[:-1] + [3, nr_mix])) # Coeffs (3 * K)
        l = tf.reshape(l[:, :, :, 4*nr_mix:], xs + [nr_mix * 2])                            # Means and scale (2 * 2 * K)
        means = l[:, :, :, :, :nr_mix]
        log_scales = tf.maximum(l[:, :, :, :, nr_mix:2*nr_mix], -7.)
        # Mean dependecies
        m1 = tf.reshape(means[:, :, :, 0, :] + coeffs[:, :, :, 0, :] * L, [xs[0], xs[1], xs[2], 1, nr_mix])
        m2 = tf.reshape(means[:, :, :, 1, :] + coeffs[:, :, :, 1, :] * L + coeffs[:, :, :, 2, :] * x[:, :, :, 0, :],  [xs[0], xs[1], xs[2], 1, nr_mix])
        means = tf.concat([m1, m2], 3)

    # Compute log likelihood
    centered_x = x - means
    inv_stdv = tf.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = tf.nn.sigmoid(min_in)
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in)        # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = - tf.nn.softplus(min_in)        # log probability for edge case of 255 (before scaling)
    cdf_delta = cdf_plus - cdf_min                          # probability for all other cases
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2. * tf.nn.softplus(mid_in) - np.log(127.5)
    # Per-channel log-likelihood for R and G
    log_probs = tf.where(x < -0.999,
                         log_cdf_plus,
                         tf.where(x > 0.999,
                                  log_one_minus_cdf_min,
                                  tf.where(cdf_delta > 1e-5,
                                            tf.log(tf.maximum(cdf_delta, 1e-12)),
                                           log_pdf_mid)
                                 )
                        )
    # Return
    log_probs = tf.reduce_sum(log_probs, 3) + log_prob_from_logits(logit_probs)
    if sum_all:
        return -tf.reduce_sum(log_sum_exp(log_probs))
    else:
        return -log_sum_exp(log_probs)

def sample_from_discretized_mix_logistic(l, nr_mix=10, colorspace="RGB", scale_var=1., **kwargs):
    """Sample from a discrete logistic mixture.

    Args:
      l: logistic mixture parametrization
      nr_mix (int): number of mixtures
      colorspace (str): target colorspace
      scale_var (float, optional): multiplicative coefficient for the variance. Defaults to 1. For instance can be set to 0 to sample from the mode of the distribution.
      x_gray (optional): grayscale input to condition on for the lab colorspace. Should be normalized inside the [-1, 1] range
     """
    ls = int_shape(l)
    logit_probs = l[:, :, :, :nr_mix]
    sel = tf.one_hot(tf.argmax(logit_probs - tf.log(-tf.log(tf.random_uniform(logit_probs.get_shape(), minval=1e-5, maxval=1. - 1e-5))), 3), depth=nr_mix, dtype=tf.float32)
    sel = tf.reshape(sel, ls[:-1] + [1, nr_mix])
    if colorspace == "RGB":
        xs = ls[:-1] + [3]
        l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix*3])

        means = tf.reduce_sum(l[:, :, :, :, :nr_mix] * sel, 4)
        log_scales = tf.maximum(tf.reduce_sum((l[:, :, :, :, nr_mix:2 * nr_mix]) * sel, 4), -7.)
        coeffs = tf.reduce_sum(tf.nn.tanh(l[:, :, :, :, 2 * nr_mix:3*nr_mix]) * sel, 4)

        u = tf.random_uniform(means.get_shape(), minval=1e-5, maxval=1. - 1e-5)
        x = means + scale_var * tf.exp(log_scales)*(tf.log(u) - tf.log(1. - u))
        x0 = tf.minimum(tf.maximum(x[:, :, :, 0], -1.), 1.)
        x1 = tf.minimum(tf.maximum(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, -1.), 1.)
        x2 = tf.minimum(tf.maximum(x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, -1.), 1.)
        return tf.concat([tf.reshape(x0, ls[:-1] + [1]), tf.reshape(x1, ls[:-1] + [1]), tf.reshape(x2, ls[:-1] + [1])], 3)

    elif colorspace == "lab":
        L = kwargs["x_gray"][:, :, :, 0]

        xs = ls[:-1] + [2]
        coeffs = tf.reduce_sum(tf.nn.tanh(tf.reshape(l[:, :, :, nr_mix:4*nr_mix], ls[:-1] + [3, nr_mix])) * sel, 4)
        l = tf.reshape(l[:, :, :, 4*nr_mix:], xs + [nr_mix * 2])

        means = tf.reduce_sum(l[:, :, :, :, :nr_mix] * sel, 4)
        log_scales = tf.maximum(tf.reduce_sum((l[:, :, :, :, nr_mix:2*nr_mix]) * sel, 4), -7.)

        u = tf.random_uniform(means.get_shape(), minval=1e-5, maxval=1. - 1e-5)
        x = means + scale_var * tf.exp(log_scales)*(tf.log(u) - tf.log(1. - u))
        c1 = tf.minimum(tf.maximum(x[:, :, :, 0] + coeffs[:, :, :, 0] * L, -1.), 1.)
        c2 = tf.minimum(tf.maximum(x[:, :, :, 1] + coeffs[:, :, :, 1] * L + coeffs[:, :, :, 2] * c1, -1.), 1.)
        return tf.concat([tf.reshape(L, ls[:-1] + [1]), tf.reshape(c1, ls[:-1] + [1]), tf.reshape(c2, ls[:-1] + [1])], 3)
