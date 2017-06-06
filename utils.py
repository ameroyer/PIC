import numpy as np
import tensorflow as tf
from skimage.color import lab2rgb, rgb2lab


class bcolors:
    """Codes for colored bash output"""
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RES = '\033[0m'

def downsample_tf(x, downsample=1):
    """Downsamples the input 4D tensor."""
    return tf.image.resize_bilinear(x, [x.get_shape().as_list()[1] // downsample, x.get_shape().as_list()[2] // downsample])

def tile_image(x_gen, tiles=None):
    """Tiled image representations.

    Args:
      x_gen: 4D array of images (n x w x h x 3)
      tiles (int pair, optional): number of rows and columns

    Returns:
      Array of tiled images (1 x W x H x 3)
    """
    n_images = x_gen.shape[0]
    if tiles is None:
        for i in range(int(np.sqrt(n_images)), 0, -1):
            if n_images % i == 0: break
        n_rows = i; n_cols = n_images // i
    else:
        n_rows, n_cols = tiles
    full = [np.hstack(x_gen[c * n_rows:(c + 1) * n_rows]) for c in range(n_cols)]
    return np.expand_dims(np.vstack(full), 0)

def get_batches(images, size):
    """Create batches from a list of input images.

    Args:
      images: Input images array s.t. axis 0 is the batch axis
      size (int): Batch size

    Returns:
      Yields the next batch for the input images
    """
    perm = np.random.permutation(len(images))
    c = 0
    while (c + 1) * size < len(images):
        yield images[perm[c * size:(c + 1) * size]]
        c += 1

def color_to_gray(x, colorspace="RGB"):
    """Grayscale the input image.

    Args:
      x: input image array (3D or 4D) (3 color channels)
      colorspace (str): colorspace of the input image. One of "RGB" or "lab"

    Returns:
      x_gray the grayscale version of x (1 color channel)
    """
    expand_func = tf.expand_dims if isinstance(x, tf.Tensor) else np.expand_dims
    if colorspace == "RGB":
        return expand_func(0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2], -1)
    elif colorspace == "lab":
        return expand_func(x[..., 0], -1)
    else:
        raise ValueError("Unknown colorspace" % colorspace)

def pcnn_norm(x, colorspace="RGB", reverse=False):
    """Normalize the input from and to [-1, 1].

    Args:
      x: input image array (3D or 4D)
      colorspace (str): Source/target colorspace, depending on the value of `reverse`
      reverse (bool, optional): If False, converts the input from the given colorspace to float in the range [-1, 1].
      Otherwise, converts the input to the valid range for the given colorspace. Defaults to False.

    Returns:
      x_norm: normalized input
    """
    if colorspace == "RGB":
        return np.cast[np.uint8](x * 127.5 + 127.5) if reverse else np.cast[np.float32]((x - 127.5) / 127.5)
    elif colorspace == "lab":
        if x.shape[-1] == 1:
            return (x * 50. + 50.) if reverse else np.cast[np.float32]((x - 50.) / 50.)
        else:
            a = np.array([50., +0.5, -0.5], dtype=np.float32)
            b = np.array([50., 127.5, 127.5], dtype=np.float32)
            return np.cast[np.float64](x * b + a) if reverse else np.cast[np.float32]((x - a) / b)
    else:
        raise ValueError("Unknown colorspace" % colorspace)

def convert_color(x, colorspace="RGB", normalized_in=False, normalized_out=False, reverse=False):
    """ convert image x from the given colorspace to RGB back and forth
    Ranges and Shapes:
     * (RGB) BxWxHx3 [0; 255]
     * (LAB) BxWxHx3 [0; 100] [-127; 128] [-128; 127]
     * (DLAB) BxWxHx2 [0; 100] [0; 313]

    Args:
      x: input image array (3D or 4D)
      colorspace (str): Source/target colorspace, depending on the value of `reverse`
      reverse (bool, optional): if False, convert from `colorspace` to RGB, else reverse conversion
      normalized_in (bool, optional): If True the input is assumed to be given in the normalized space [-1, 1], otherwise in the valid range of the source colorspace
      normalized_out (bool, optional): If True, the output will be sent to [-1, 1], otherwise to the valid range of the target colorspace

    Returns:
      x_norm: normalized input
    """
    if colorspace == "RGB":
        return pcnn_norm(x) if (not normalized_in and normalized_out) else pcnn_norm(x, reverse=True) if (normalized_in and not normalized_out) else x
    elif colorspace == "lab":
        y = pcnn_norm(x, colorspace="lab" if reverse else "RGB", reverse=True) if normalized_in else x
        y = lab_to_rgb(y) if reverse else rgb_to_lab(y)
        y = pcnn_norm(y, colorspace="RGB" if reverse else "lab") if normalized_out else y
        return y
    else:
        raise ValueError("Unknown colorspace" % colorspace)

def rgb_to_lab(x):
    """Converts RGB image to the lab colorspace [0; 100] [-127; 128] [-128; 127]."""
    return rgb2lab(x)

def lab_to_rgb(x, eps=1e-8):
    """Converts a lab image [0; 100] [-127; 128] [-128; 127] to a valid RGB image."""
    x_rectified = np.array(x)
    upper_bound = 200 * (x[..., 0] + 16.) / 116. - eps
    x_rectified[..., 2] = np.clip(x_rectified[..., 2], - float('inf'), upper_bound)
    return np.array([lab2rgb(y) * 255. for y in x_rectified]).astype(np.uint8)
