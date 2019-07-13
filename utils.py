import numpy as np
import os
from glob import glob
from ops import lerp

import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2

class ImageData:

    def __init__(self, img_size):
        self.img_size = img_size
        self.channels = 3

    def image_processing(self, filename):
        x = tf.read_file(filename)
        img = tf.image.decode_jpeg(x, channels=self.channels, dct_method='INTEGER_ACCURATE')
        img = preprocess_fit_train_image(img, self.img_size)

        return img

def adjust_dynamic_range(images):
    drange_in = [0.0, 255.0]
    drange_out = [-1.0, 1.0]
    scale = (drange_out[1] - drange_out[0]) / (drange_in[1] - drange_in[0])
    bias = drange_out[0] - drange_in[0] * scale
    images = images * scale + bias
    return images


def random_flip_left_right(images):
    s = tf.shape(images)
    mask = tf.random_uniform([1, 1, 1], 0.0, 1.0)
    mask = tf.tile(mask, [s[0], s[1], s[2]]) # [h, w, c]
    images = tf.where(mask < 0.5, images, tf.reverse(images, axis=[1]))
    return images


def smooth_crossfade(images, alpha):
    s = tf.shape(images)
    y = tf.reshape(images, [-1, s[1] // 2, 2, s[2] // 2, 2, s[3]])
    y = tf.reduce_mean(y, axis=[2, 4], keepdims=True)
    y = tf.tile(y, [1, 1, 2, 1, 2, 1])
    y = tf.reshape(y, [-1, s[1], s[2], s[3]])
    images = lerp(images, y, alpha)
    return images

def preprocess_fit_train_image(images, res):
    images = tf.image.resize(images, size=[res, res], method=tf.image.ResizeMethod.BILINEAR)
    images = adjust_dynamic_range(images)
    images = random_flip_left_right(images)

    return images

def load_data(dataset_name) :
    x = glob(os.path.join("./dataset", dataset_name, '*.*'))

    return x

def save_images(images, size, image_path):
    # return imsave(inverse_transform(images), size, image_path)
    return imsave(images, size, image_path)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

def imsave(images, size, path):
    # return scipy.misc.imsave(path, merge(images, size))

    images = merge(images, size)
    images = post_process_generator_output(images)
    images = cv2.cvtColor(images.astype('uint8'), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, images)

def inverse_transform(images):
    return (images+1.)/2.

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def str2bool(x):
    return x.lower() in ('true')

def get_checkpoint_res(checkpoint_counter, batch_sizes, iteration, start_res, end_res, gpu_num, end_iteration, do_trans) :
    batch_sizes_key = list(batch_sizes.keys())

    start_index = batch_sizes_key.index(start_res)

    iteration_per_res = []

    for res, bs in batch_sizes.items() :

        if do_trans[res] :
            if res == end_res :
                iteration_per_res.append(end_iteration // (bs * gpu_num))
            else :
                iteration_per_res.append(iteration // (bs * gpu_num))
        else :
            iteration_per_res.append((iteration // 2) // (bs * gpu_num))

    iteration_per_res = iteration_per_res[start_index:]

    for i in range(len(iteration_per_res)) :

        checkpoint_counter = checkpoint_counter - iteration_per_res[i]

        if checkpoint_counter < 1 :
            return i+start_index

def post_process_generator_output(generator_output):

    drange_min, drange_max = -1.0, 1.0
    scale = 255.0 / (drange_max - drange_min)

    scaled_image = generator_output * scale + (0.5 - drange_min * scale)
    scaled_image = np.clip(scaled_image, 0, 255)

    return scaled_image