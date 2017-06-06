import tensorflow as tf
import nn


def EmbeddingCIFAR(inp):
    """Returns the CIFAR-specific grayscale embedding for the given input."""
    with tf.name_scope("embedding"):
        channels_cond = 32
        leak = nn.conv(inp, "conv_leak", filter_size=(3, 3), stride=1, out_channels=channels_cond)
        with tf.name_scope("down_pass"):
            leak = nn.gated_resnet(leak, "down_leak_%d" % 1, a=None, conv=nn.conv)
            leak = nn.gated_resnet(leak, "down_leak_%d" % 2, a=None, conv=nn.conv)
            channels_cond *= 2
            leak = nn.conv(leak, "downscale_leak_1", filter_size=(3, 3), stride=2, out_channels=channels_cond)

            leak = nn.gated_resnet(leak, "down_leak_%d" % 3, a=None, conv=nn.conv)
            leak = nn.gated_resnet(leak, "down_leak_%d" % 4, a=None, conv=nn.conv)
            channels_cond *= 2
            leak = nn.conv(leak, "downscale_leak_2", filter_size=(3, 3), stride=1, out_channels=channels_cond)

            leak = nn.gated_resnet(leak, "down_leak_%d" % 5, a=None, conv=nn.conv)
            leak = nn.gated_resnet(leak, "down_leak_%d" % 6, a=None, conv=nn.conv)
            channels_cond *= 2
            leak = nn.conv(leak, "downscale_leak_3", filter_size=(3, 3), stride=1, out_channels=channels_cond)

            leak = nn.gated_resnet(leak, "down_leak_%d" % 7, a=None, conv=nn.conv, dilation=2)
            leak = nn.gated_resnet(leak, "down_leak_%d" % 8, a=None, conv=nn.conv, dilation=2)
            leak = nn.gated_resnet(leak, "down_leak_%d" % 9, a=None, conv=nn.conv, dilation=2)
            embedding = nn.conv(leak, "downscale_leak_4", filter_size=(3, 3), stride=1, out_channels=channels_cond)

    return embedding

def EmbeddingImagenet(inp):
    """Returns the Imagenet-specific grayscale embedding for the given input."""
    with tf.name_scope("embedding"):
        channels_cond = 64
        leak = nn.conv(inp, "conv_leak", filter_size=(3, 3), stride=1, out_channels=channels_cond)
        with tf.name_scope("down_pass"):
            leak = nn.gated_resnet(leak, "down_leak_%d" % 1, a=None, conv=nn.conv)
            leak = nn.gated_resnet(leak, "down_leak_%d" % 2, a=None, conv=nn.conv)
            channels_cond *= 2
            leak = nn.conv(leak, "downscale_leak_1", filter_size=(3, 3), stride=2, out_channels=channels_cond)

            leak = nn.gated_resnet(leak, "down_leak_%d" % 3, a=None, conv=nn.conv)
            leak = nn.gated_resnet(leak, "down_leak_%d" % 4, a=None, conv=nn.conv)
            channels_cond *= 2
            leak = nn.conv(leak, "downscale_leak_2", filter_size=(3, 3), stride=2, out_channels=channels_cond)

            leak = nn.gated_resnet(leak, "down_leak_%d" % 5, a=None, conv=nn.conv)
            leak = nn.gated_resnet(leak, "down_leak_%d" % 6, a=None, conv=nn.conv)
            channels_cond *= 2
            leak = nn.conv(leak, "downscale_leak_3", filter_size=(3, 3), stride=1, out_channels=channels_cond)

            leak = nn.gated_resnet(leak, "down_leak_%d" % 7, a=None, conv=nn.conv, dilation=2)
            leak = nn.gated_resnet(leak, "down_leak_%d" % 8, a=None, conv=nn.conv, dilation=2)
            leak = nn.gated_resnet(leak, "down_leak_%d" % 9, a=None, conv=nn.conv, dilation=2)
            leak = nn.conv(leak, "downscale_leak_4", filter_size=(3, 3), stride=1, out_channels=channels_cond)

            leak = nn.gated_resnet(leak, "down_leak_%d" % 10, a=None, conv=nn.conv, dilation=4)
            leak = nn.gated_resnet(leak, "down_leak_%d" % 11, a=None, conv=nn.conv, dilation=4)
            leak = nn.gated_resnet(leak, "down_leak_%d" % 12, a=None, conv=nn.conv, dilation=4)

            # Minor bug: wrong number of channels (TODO: retrian the model and fix the code)
            embedding = nn.conv(leak, "downscale_leak_5", filter_size=(3, 3), stride=1, out_channels=160)

    return embedding

def PIColorization(x, x_gray, channels, l, num_outputs, dataset, return_embedding=False):
    """Define the auto-regressive network.
    Args:
      x: input
      x_gray: grayscale embedding
      channels: network width
      l (int): number of residual layers in the embedding network
      num_outputs (int): number of coeffs (ie logistic mixtures * n_coeffs per mixture)
      dataset (str): dataset
      return_embedding (bool, optional): if True, also return the embedding. Defaults to False
    """
    # PIC
    with tf.name_scope("pic"):

        with tf.name_scope("pad"):
            x_pad = tf.concat([x, tf.ones(nn.int_shape(x)[:-1] + [1])], 3, name="x_pad")
            x_gray = tf.concat([x_gray, tf.ones(nn.int_shape(x_gray)[:-1] + [1])], 3, name="gray_pad")

        # Embedding
        assert(dataset in ['cifar', 'imagenet'])

        if dataset == 'cifar':
            embedding = EmbeddingCIFAR(x_gray)
        elif dataset == 'imagenet':
            embedding = EmbeddingImagenet(x_gray)

        # PixelCNN++
        with tf.name_scope("pcnn"):
            u = nn.down_shift(nn.down_shifted_conv2d(x_pad, "conv_down", filter_size=(2, 3), out_channels=channels))
            ul = nn.down_shift(nn.down_shifted_conv2d(x_pad, "conv_down_2",  filter_size=(1, 3), out_channels=channels)) + \
                nn.right_shift(nn.down_right_shifted_conv2d(x_pad, "conv_down_right", filter_size=(2, 1), out_channels=channels))

            for rep in range(l):
                u = nn.gated_resnet(u, "shortrange_down_%d" % rep, a=embedding, conv=nn.down_shifted_conv2d)
                ul = nn.gated_resnet(ul, "shortrange_down_right_%d" % rep, a=tf.concat([u, embedding], 3), conv=nn.down_right_shifted_conv2d)

        x_out = nn.conv(tf.nn.elu(ul), "conv_last", (1, 1), num_outputs)

    if return_embedding:
        return x_out, embedding
    else:
        return x_out
