from __future__ import print_function
import os
import sys
import argparse
from time import gmtime, strftime

import h5py
import numpy as np
import tensorflow as tf
import scipy.ndimage as nd
from scipy.misc import imread, imresize, imsave

import nn
import models
from utils import *

is_python3 = (sys.version_info[0] == 3)
if is_python3:
    from pickle import load as pickle_load
else:
    import cPickle
    def pickle_load(file, **kwargs):
        return cPickle.load(file)
    range = xrange


### Set command line parameters
parser = argparse.ArgumentParser()
parser.add_argument('-in', '--data_dir', type=str, help='Path to the dataset directory')
parser.add_argument('-log', '--log_dir', type=str, default='./log', help='Path to the log directory')
parser.add_argument('--mode', type=str, default='train', help='Mode. One of train, eval or demo')
parser.add_argument('--input', type=str, default=None, help='Path to input grayscale image for the demo mode.')
parser.add_argument('-w', '--model', type=str, default='', help='Path to an (optional) pre-trained model checkpoint')
parser.add_argument('--color', type=str, default='lab',  help='Colorspace. One of RGB or lab')
parser.add_argument('--dataset', type=str, default='cifar',  help='Image dataset')

# PixelCNN
parser.add_argument('--sample_mode', dest='sample_mode', action='store_true', help='If True, generate samples corresponding to the mode of the PIC distribution along the random ones from PIC')
parser.add_argument('--sample_embedding', dest='sample_embedding', action='store_true', help='If True, generate samples from the feed-forward embedding network along the ones from PIC')
parser.add_argument('-l', '--nr_pxpp_blocks', type=int, default=2, help='Number of PixelCNN blocks')
parser.add_argument('-c', '--nr_channels', type=int, default=160, help='Width of the PixelCNN network')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10, help='Number of logistic components in the mixture.')

# Hyperparameters
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('-ld', '--lr_decay', type=float, default=0.99999, help='Learning rate decay, applied every decay epoch')

parser.add_argument('-g', '--nr_gpus', type=int, default=1, help='Number of GPUs')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size per GPU (train)')
parser.add_argument('-bt', '--test_batch_size', type=int, default=16, help='Batch size per GPU (test)')
parser.add_argument('-bin', '--init_batch_size', type=int, default=300, help='Batch size per GPU for the data-dependent initialization.')

parser.add_argument('-d', '--dropout', type=float, default=0.5, help='Dropout.')
parser.add_argument('-r', '--reg_weight', type=float, default=0., help='Regularization weight.')
parser.add_argument('-p', '--polyak_decay', type=float, default=0.9995, help='Polyak averaging decay')
parser.add_argument('-ds', '--downsample', type=int, default=2, help='Rate for chroma downsampling')

# Epochs
parser.add_argument('-n', '--epochs', type=int, default=150, help='Number of epochs')
parser.add_argument('-ns', '--save_epochs', type=int, default=5, help='Model checkpoint saving interval')
parser.add_argument('-ng', '--gen_epochs', type=int, default=1, help='Sample generation interval')
parser.add_argument('-ngen', '--n_generations', type=int, default=1, help='Number of repetitions per sampling experiments')
parser.add_argument('-nt', '--test_epochs', type=int, default=1, help='Log-likelhood on the validation set computation interval')
args = parser.parse_args()

### Init
if args.color not in ['RGB', 'lab']:
    raise ValueError("Unknown color mode", args.color)

if args.mode not in ['train', 'eval', 'demo']:
    raise ValueError("Unknown excution mode", args.mode)

if args.mode in ['eval', 'demo'] and not args.model:
    raise IOError("Error: missing pretrained file in eval mode")

if args.mode == 'demo':
    if not args.input:
        raise IOError("Error: missing input grayscale image in demo mode")
    args.test_batch_size = 1
    
num_outputs = (10 * args.nr_logistic_mix if args.color == "RGB"
               else 8 * args.nr_logistic_mix if args.color == "lab"
               else 0)

out_name = strftime("%m-%d_%H-%M", gmtime())
out_path = "model_%s.pkl" % out_name
log_dir = os.path.join(args.log_dir, out_name)

### Load data
if args.mode != 'demo':
    if args.dataset == 'cifar':
        if is_python3:
            TRAIN = pickle_load(open(os.path.join(args.data_dir, 'train'), 'rb'), encoding='latin1')
            TEST = pickle_load(open(os.path.join(args.data_dir, 'test'), 'rb'), encoding='latin1')
        else:
            TRAIN = pickle_load(open(os.path.join(args.data_dir, 'train')))
            TEST = pickle_load(open(os.path.join(args.data_dir, 'test')))
        images_train = TRAIN['data'].reshape([-1, 3, 32, 32]).transpose([0, 2, 3, 1])
        images_test = TEST['data'].reshape([-1, 3, 32, 32]).transpose([0, 2, 3, 1])
    elif args.dataset == 'imagenet':
        with h5py.File(os.path.join(args.data_dir, 'imagenet-128.hdf5')) as f:
            images_train = np.array(f['train'])
            images_test = np.array(f['val'])
    else:
        raise NameError('Unknown dataset')

    WIDTH, HEIGHT, C_IN = images_train.shape[1:]   # number of channel of the input RGB
    np.random.shuffle(images_test)
    images_test_gen = images_test[:args.test_batch_size * args.nr_gpus]
else:
    if args.dataset == 'cifar':
        WIDTH, HEIGHT, C_IN = 32, 32, 3
    elif args.dataset == 'imagenet':
        WIDTH, HEIGHT, C_IN = 128, 128, 3
    else:
        raise NameError('Unknown dataset')

### Model template
def picolor(x, x_gray, return_embedding=False):
    global args, num_outputs
    """PIC model.

    Args:
      x: colored output (b x w x h x 3), normalized between -1 and 1
      x_gray: gray input (b x w x h x 1), normalized between -1 and 1
      return_embedding: If True, additionally returns the grayscale embedding

    Returns:
      l: PIC output
    """
    pic_out = models.PIColorization(x, x_gray, args.nr_channels, args.nr_pxpp_blocks, num_outputs,
                                       dataset=args.dataset, return_embedding=return_embedding)
    return pic_out

picolor_with_scope = tf.make_template('picolor_template', picolor)


############### Model (init)
tf.GLOBAL = {}
tf.GLOBAL['dropout'] = 0.
tf.GLOBAL['phase'] = 'train'
with tf.name_scope("initialization"):
    tf.GLOBAL['init'] = True
    tf.GLOBAL['reg'] = args.reg_weight
    with tf.name_scope("x_init"):
        x_init = tf.placeholder(shape=(args.init_batch_size, WIDTH, HEIGHT, C_IN),
                                dtype=tf.float32, name="x_init")          # rgb input
        x_init_clr = tf.placeholder(shape=(args.init_batch_size, WIDTH, HEIGHT, C_IN),
                                    dtype=tf.float32, name="x_init_clr")  # colorspace converted input
        x_init_gray = color_to_gray(x_init_clr, colorspace=args.color)    # grayscale input
    with tf.name_scope('picolor_init'):
        with tf.device('/cpu:0'):
            picolor_with_scope(downsample_tf(x_init, args.downsample),
                               x_init_gray)        # PIC output
    tf.GLOBAL['init'] = False

all_params = tf.trainable_variables()
ema = tf.train.ExponentialMovingAverage(decay=args.polyak_decay)
maintain_averages_op = tf.group(ema.apply(all_params))


############### Model (training)
tf.GLOBAL['dropout'] = args.dropout
with tf.name_scope("input"):
    x = tf.placeholder(shape=(args.batch_size * args.nr_gpus, WIDTH, HEIGHT, C_IN),
                       dtype=tf.float32, name="x")          # input x (RGB + normalized)
    x_clr = tf.placeholder(shape=(args.batch_size * args.nr_gpus, WIDTH, HEIGHT, C_IN),
                           dtype=tf.float32, name="x_clr")  # input x (colorspace + normalized)
    cond = tf.less(tf.random_uniform([], 0, 1), 0.5)
    x_aug = tf.cond(cond, lambda: x, lambda: tf.reverse(x, [2]))
    xs = tf.split(x_aug, args.nr_gpus, 0)
    x_clr_aug = tf.cond(cond, lambda: x_clr, lambda: tf.reverse(x_clr, [2]))
    xs_clr = tf.split(x_clr_aug, args.nr_gpus, 0)

with tf.name_scope('MAIN'):
    for i in range(args.nr_gpus):
        x_in = tf.identity(xs[i], name="color_image")            # x_in RGB (BxWxHx3)
        x_gray = color_to_gray(xs_clr[i], colorspace=args.color) # x_gray colorspace (BxWxHx1)

        with tf.device('/gpu:%i' % i):
            picolor_out, embedding_out = picolor_with_scope(downsample_tf(x_in, args.downsample),
                                                            x_gray,
                                                            return_embedding=True)

            tf.GLOBAL['dropout'] = 0.0
            tf.GLOBAL['ema'] = ema
            picolor_out_val = picolor_with_scope(downsample_tf(x_in, args.downsample), x_gray)
            tf.GLOBAL['dropout'] = args.dropout
            tf.GLOBAL.pop('ema')

        with tf.name_scope("loss_gen"):
            loss_gen = nn.colorization_loss(downsample_tf(xs_clr[i], args.downsample),
                                            picolor_out,
                                            nr_mix=args.nr_logistic_mix,
                                            colorspace=args.color)
            tf.add_to_collection('total_loss_gen', loss_gen)

            loss_embedding = nn.colorization_loss(downsample_tf(xs_clr[i], args.downsample),
                                                  embedding_out[..., :num_outputs],
                                                  nr_mix=args.nr_logistic_mix,
                                                  colorspace=args.color)
            tf.add_to_collection('total_loss_embedding', loss_embedding)

            loss_gen_val = nn.colorization_loss(downsample_tf(xs_clr[i], args.downsample),
                                                picolor_out_val,
                                                nr_mix=args.nr_logistic_mix,
                                                colorspace=args.color)
            tf.add_to_collection('total_loss_gen_val', loss_gen_val)

# tensorboard
with tf.name_scope("bpd"):
    l2_reg = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    bits_per_dim = tf.add_n(tf.get_collection('total_loss_gen')) / float(args.batch_size * args.nr_gpus)
    bits_per_dim_embedding = tf.add_n(tf.get_collection('total_loss_embedding')) / float(args.batch_size * args.nr_gpus)
    bits_per_dim_val = tf.add_n(tf.get_collection('total_loss_gen_val')) / float(args.batch_size * args.nr_gpus)
sum_bpd = tf.summary.scalar("train_bits_per_dimension", bits_per_dim)
sum_bpd_embedding = tf.summary.scalar("train_bits_per_dimension_embedding", bits_per_dim_embedding)

# train
learning_rate = tf.placeholder(tf.float32, shape=[], name="lr")
opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.95, beta2=0.9995, epsilon=1e-06)
train_step = opt.minimize(bits_per_dim + bits_per_dim_embedding + l2_reg, colocate_gradients_with_ops=True)


############### Model (sampling)
    # Differences with train time
    # no dependency between x and x_gray
    # Both inputs are expected to be already normalized
with tf.name_scope("sampling"):
    tf.GLOBAL['dropout'] = 0
    tf.GLOBAL['phase'] = 'test'
    tf.GLOBAL['ema'] = ema
    samplers_from_pic = [0] * args.nr_gpus
    samplers_from_embedding = [0] * args.nr_gpus
    # scale_var: 1 usually, 0 when sampling from the mode
    scale_var = tf.placeholder(tf.float32, shape=[], name="scale_var")

    with tf.name_scope("x_gray"):
        # Gray image input (colorspace + normalized)
        x_gray_gen = tf.placeholder(shape=(args.test_batch_size * args.nr_gpus, WIDTH, HEIGHT, 1),
                                    dtype=tf.float32, name="x_gray_gen")
        x_gray_gens = tf.split(x_gray_gen, args.nr_gpus, 0)

    with tf.name_scope("canvas"):
        # past PIC predictions, in [-1, 1]
        x_canvas_gen = tf.placeholder(shape=(args.test_batch_size * args.nr_gpus,
                                             WIDTH // args.downsample,
                                             HEIGHT // args.downsample,
                                             C_IN),
                                       dtype=tf.float32, name="x_canvas_gen")
        x_canvas_gens = tf.split(x_canvas_gen, args.nr_gpus, 0)

    embedding_cache = [None] * args.nr_gpus
    with tf.name_scope('picolor_sampling'):
        for i in range(args.nr_gpus):
            with tf.device('/gpu:%d' % i):
                picolor_out, embedding_cache[i] = picolor_with_scope(x_canvas_gens[i],
                                                                     x_gray_gens[i],
                                                                     return_embedding=True)
                samplers_from_pic[i] = nn.sample_from_discretized_mix_logistic(picolor_out,
                                                                               nr_mix=args.nr_logistic_mix,
                                                                               colorspace=args.color,
                                                                               scale_var=scale_var,
                                                                               x_gray=downsample_tf(x_gray_gens[i], args.downsample))
                samplers_from_embedding[i] = nn.sample_from_discretized_mix_logistic(embedding_cache[i][..., :num_outputs],
                                                                                     nr_mix=args.nr_logistic_mix,
                                                                                     colorspace=args.color,
                                                                                     scale_var=scale_var,
                                                                                     x_gray=downsample_tf(x_gray_gens[i], args.downsample))
    tf.GLOBAL.pop('ema')

# tensorboard dummies for @generate_samples
# Reconstruction
bpd_rec =  tf.get_variable("bpd_rec", initializer=tf.constant(0, dtype=tf.float64))
sum_bpd_rec = tf.summary.scalar("rec_bits_per_dimension", bpd_rec)

rec_error =  tf.get_variable("rec_error", initializer=tf.constant(0, dtype=tf.float64))
sum_rec_error = tf.summary.scalar("rec_error", rec_error)

# Generation
gen_error =  tf.get_variable("gen_error", initializer=tf.constant(0, dtype=tf.float64))
sum_gen_error = tf.summary.scalar("gen_error", gen_error)

gen_error_gray =  tf.get_variable("gen_error_gray", initializer=tf.constant(0, dtype=tf.float64))
sum_gen_error_gray = tf.summary.scalar("gen_error_gray", gen_error_gray)


############### Generate samples
def generate_samples(images, sess, summary_writer, reconstruct=False, from_embedding=False, sample_mode=False, id=0, resolution=1):
    """ Generate samples from a given set of gray images, starting from a blank canvas or original color images (if ``reconstruct`` is True)

    Args:
      images: input ground-truth color images
      sess: tf session
      summary_writer: tf summary writer object
      reconstruct: if True, generate reconstructions, i.e. the ground truth color image is given as additional input to the auto-regressive component
      from_embedding: if True, sample from the feed-forward embedding network, else from PIC
      sample_mode: if True, sample from the mode of the distribution
      id (int): identifier
      resolution (int): chroma sampling resolution
    """
    global WIDTH, HEIGHT, C_IN, args, samplers_from_pic, samplers_from_embedding
    global rec_error, rec_error_gray, bpd_gen, gen_error, gen_error_gray
    gray_images = color_to_gray(convert_color(images, colorspace=args.color, normalized_out=True), colorspace=args.color) #BxWxHx1

    samplers = samplers_from_embedding if from_embedding else samplers_from_pic
    feed = ({x_gray_gen: gray_images, scale_var:float(not sample_mode)})

    if reconstruct:
        feed.update({x_canvas_gen: nd.zoom(pcnn_norm(images), (1.0, 1.0 / args.downsample, 1.0 / args.downsample, 1.0), order=1)})
        new_x_gen_np = sess.run(samplers, feed)
        x_gen = convert_color(np.concatenate(new_x_gen_np, axis=0),
                              colorspace=args.color,
                              normalized_in=True,
                              normalized_out=True,
                              reverse=True)
    else:
        x_gen = np.zeros((args.test_batch_size * args.nr_gpus, WIDTH // args.downsample, HEIGHT // args.downsample, C_IN), dtype=float) #RGB canvas
        feed.update({embedding_cache[i]: sess.run(embedding_cache[i], {x_gray_gen: gray_images}) for i in range(args.nr_gpus)})

        for yi in range(0, WIDTH // args.downsample):
            for xi in range(0, HEIGHT // args.downsample):
                feed.update({x_canvas_gen: x_gen})
                new_x_gen_np = np.concatenate(sess.run(samplers, feed))
                x_gen[:, yi, xi, :] = convert_color(new_x_gen_np,
                                                    colorspace=args.color,
                                                    normalized_in=True,
                                                    normalized_out=True,
                                                    reverse=True)[:, yi, xi, :]
    if args.color == 'RGB':
        x_gen = nd.zoom(x_gen, (1.0, args.downsample, args.downsample, 1.0), order=1)
    else:
        x_gen = nd.zoom(convert_color(x_gen,
                                      colorspace=args.color,
                                      normalized_in=True,
                                      normalized_out=True,
                                      reverse=False)[..., 1:],
                                      (1.0, args.downsample, args.downsample, 1.0), order=1)
        x_gen = np.concatenate([gray_images, x_gen], axis=3)
    x_gen = convert_color(x_gen, colorspace=args.color,
                          normalized_in=True,
                          normalized_out=False,
                          reverse=True)

    # Summary
    imgs = tile_image(x_gen)
    if reconstruct:
        summary_str = sess.run(tf.summary.image("rec%s_%d" % ("embd" if from_embedding else "pic", id), imgs, max_outputs=1))
    else:
        summary_str = sess.run(tf.summary.image("gen%s_%d%s" % ("embd" if from_embedding else "pic", id, '_mode'  if sample_mode else ''), imgs, max_outputs=1))
    summary_writer.add_summary(summary_str, id)


############### Main
inits = tf.global_variables_initializer()
train_summary_op = tf.summary.merge([sum_bpd, sum_bpd_embedding])
rec_summary_op = tf.summary.merge([sum_bpd_rec, sum_rec_error])
gen_summary_op = tf.summary.merge([sum_gen_error, sum_gen_error_gray])

with tf.Session() as sess:
    ### Init saver and summary objects
    print()
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    else:
        for f in os.listdir(log_dir):
            file_path = os.path.join(log_dir, f)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)
    summary_writer = tf.summary.FileWriter(log_dir, graph=sess.graph)
    saver = tf.train.Saver()

    if args.model:
        print("Loading model from", "%s%s%s" % (bcolors.YELLOW, args.model, bcolors.RES), "...")
        saver.restore(sess, args.model)
    else:
        print(bcolors.YELLOW, "Initializing model ...", bcolors.RES)
        xx = images_train[:args.init_batch_size]
        sess.run(inits, {x_init: convert_color(xx,
                                               colorspace="RGB",
                                               normalized_out=True),
                         x_init_clr: convert_color(xx,
                                                   colorspace=args.color,
                                                   normalized_out=True)})

    if args.n_generations > 0 and args.mode in ['train', 'test']:
        summary_str = sess.run(tf.summary.image("original", tile_image(images_test_gen), max_outputs=1))
        summary_writer.add_summary(summary_str)

    ### Training mode
    lr = args.learning_rate
    if args.mode == 'train':
        try:
            for i in range(1, args.epochs + 1):
                ### Train
                avg_bpd = 0
                b = 0
                for xx in get_batches(images_train, args.nr_gpus * args.batch_size):
                    _, bpd, summary_str, _ = sess.run([train_step, bits_per_dim, train_summary_op, maintain_averages_op],
                                                      {x: convert_color(xx,
                                                                        colorspace="RGB",
                                                                        normalized_out=True),
                                                       x_clr: convert_color(xx,
                                                                            colorspace=args.color,
                                                                            normalized_out=True),
                                                       learning_rate: lr})
                    lr *= args.lr_decay
                    avg_bpd += bpd
                    b += 1
                    summary_writer.add_summary(summary_str, (i - 1) * len(images_train) + b * args.batch_size * args.nr_gpus)
                    print("\rBatch: %d/%d (bpd:" % (b, len(images_train) // (args.batch_size * args.nr_gpus)),
                          "%s%.4f%s)" % (bcolors.RED, bpd, bcolors.RES),
                          "(avg_bpd: %.3f)" % (avg_bpd / b),
                          end='')

                ### Validation
                avg_bpd_val = 0
                b_val = 0
                if args.test_epochs > 0 and not i % args.test_epochs:
                    for xx in get_batches(images_test, args.nr_gpus * args.batch_size):
                        bpd = sess.run(bits_per_dim_val, {x: convert_color(xx,
                                                                           colorspace="RGB",
                                                                           malized_out=True),
                                                          x_clr: convert_color(xx,
                                                                               colorspace=args.color,
                                                                               normalized_out=True)})
                        avg_bpd_val += bpd
                        b_val += 1
                        print("\r(val) Batch: %d/%d" % (b_val, len(images_test) // (args.batch_size * args.nr_gpus)), ' ' * 35, end = '')
                    _ = sess.run(bpd_rec.assign(avg_bpd_val / b_val))

                ### Sampling experiments
                if args.gen_epochs > 0 and not i % args.gen_epochs:
                    print("\r(val) Sampling experiments...", ' ' * 10, end='')
                    if args.sample_embedding:
                        generate_samples(images_test_gen, sess, summary_writer, reconstruct=True, from_embedding=True, id=i)
                        generate_samples(images_test_gen, sess, summary_writer, from_embedding=True, id=i, resolution=1)

                    if args.sample_mode:
                        generate_samples(images_test_gen, sess, summary_writer, sample_mode=True, id=i)

                    generate_samples(images_test_gen, sess, summary_writer, reconstruct=True, id=i)
                    for ngen in range(args.n_generations):
                        generate_samples(images_test_gen, sess, summary_writer, id=i, resolution=1)

                ### Output Epoch Summary
                print("\repoch %s%d/%d%s %s:" % (bcolors.CYAN, i, args.epochs, bcolors.RES, strftime("%m-%d_%H-%M", gmtime())),
                      "(avg_bpd:", "%s%.3f%s)" % (bcolors.YELLOW, avg_bpd / b, bcolors.RES),
                      "" if b_val <= 0 else "(avg_bpd_val: {}{:.3f}{})".format(bcolors.YELLOW, avg_bpd_val / b_val, bcolors.RES),
                      ' ' * 30)
                summary_writer.add_summary(sess.run(rec_summary_op), i)
                summary_writer.add_summary(sess.run(gen_summary_op), i)

                if args.save_epochs > 0 and not i % args.save_epochs:
                    saver.save(sess, os.path.join(log_dir, "model.ckpt"))
                    print("%sSaved model at epoch %d !%s" % (bcolors.RED, i, bcolors.RES))

            print("%sEnd%s" % (bcolors.CYAN, bcolors.RES))

        except KeyboardInterrupt:
            print("Exiting at epoch %d/%d:" % (i, args.epochs))
            print("Last saved epoch=%d in %s" % (i - 1 - (i - 1) % args.save_epochs, out_path))

    ### Evaluation mode
    elif args.mode == 'eval':
        print("Validation bits per dimension...", ' ' * 10)
        avg_bpd_val = 0
        b_val = 0
        for xx in get_batches(images_test, args.nr_gpus * args.batch_size):
            bpd = sess.run(bits_per_dim_val, {x: convert_color(xx, colorspace="RGB", normalized_out=True),
                                              x_clr: convert_color(xx, colorspace=args.color, normalized_out=True)})
            avg_bpd_val += bpd
            b_val += 1
            print("\r(val) Batch: %d/%d" % (b_val, len(images_test) // (args.batch_size * args.nr_gpus)), ' ' * 35, end='')

        print("\rValidation score: %s%.3f%s" % (bcolors.YELLOW, avg_bpd_val / b_val, bcolors.RES))

        ### Sampling experiments
        print("Sampling experiments...")
        print("Reconstruction ...", end='')
        generate_samples(images_test_gen, sess, summary_writer, reconstruct=True, id=0)
        print("\rReconstruction ...", bcolors.CYAN, "Done", bcolors.RES)

        if args.sample_mode:
            print("Sample from PIC mode ...", end='')
            generate_samples(images_test_gen, sess, summary_writer, sample_mode=True)
            print("\rSample from PIC mode ...", bcolors.CYAN, "Done", bcolors.RES)
        # Sample
        for ngen in range(args.n_generations):
            print("Sample %d ..." % (ngen + 1), end='')
            generate_samples(images_test_gen, sess, summary_writer, id=(ngen + 1))
            generate_samples(images_test_gen, sess, summary_writer, from_embedding=True, id=(ngen + 1))
            print("\rSample %d ..." % (ngen + 1), bcolors.CYAN, "Done", bcolors.RES)

    ### Apply the model on one image
    else:
        image = imread(os.path.abspath(args.input))
        w, h, _ = image.shape
        image = imresize(image, (WIDTH, HEIGHT))[:, :, 0][:, :, None]
        base, ext = os.path.basename(args.input).rsplit('.', 1)
        out_path = os.path.join(log_dir, "%s_colorized.%s" % (base, ext))

        # Sampler
        image = (image - 127.5) / 127.5
        image = image.astype(np.float32)
        image = image[None, :, :, :]

        x_gen = np.zeros((1, WIDTH // args.downsample, HEIGHT // args.downsample, 3), dtype=float)
        feed = ({x_gray_gen: image, scale_var:float(not args.sample_mode)})
        feed.update({embedding_cache[i]: sess.run(embedding_cache[i], {x_gray_gen: image}) for i in range(args.nr_gpus)})

        for yi in range(0, WIDTH // args.downsample):
            for xi in range(0, HEIGHT // args.downsample):
                feed.update({x_canvas_gen: x_gen})
                new_x_gen_np = np.concatenate(sess.run(samplers_from_pic, feed))
                x_gen[:, yi, xi, :] = convert_color(new_x_gen_np,
                                                    colorspace=args.color,
                                                    normalized_in=True,
                                                    normalized_out=True,
                                                    reverse=True)[:, yi, xi, :]
        if args.color == 'RGB':
            x_gen = nd.zoom(x_gen, (1.0, args.downsample, args.downsample, 1.0), order=1)
        else:
            x_gen = nd.zoom(convert_color(x_gen,
                                          colorspace=args.color,
                                          normalized_in=True,
                                          normalized_out=True,
                                          reverse=False)[..., 1:],
                                          (1.0, args.downsample, args.downsample, 1.0), order=1)
            x_gen = np.concatenate([image, x_gen], axis=3)
        x_gen = convert_color(x_gen, colorspace=args.color,
                              normalized_in=True,
                              normalized_out=False,
                              reverse=True)[0, ...]
        x_gen = imresize(x_gen, (w, h))
        print(out_path)
        imsave(out_path, x_gen)
