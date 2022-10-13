# Utility

import numpy as np

from os import listdir, mkdir, sep
from os.path import join, exists, splitext
from imageio import imread, imsave
import cv2
import skimage
import skimage.io
import skimage.transform
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PIL import Image
from functools import reduce

def list_images(directory):
    images = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        elif name.endswith('.bmp'):
            images.append(join(directory, file))
        elif name.endswith('.tiff'):
            images.append(join(directory, file))
    return images


# read images
def get_image(path, height=320, width=320, set_mode='RGB'):
    image= imread(path, pilmode=set_mode)
    #image = image.astype('uint8')
    if height is not None and width is not None:
        image = cv2.resize(image, (height, width), cv2.INTER_NEAREST)
    return image


def get_train_images(paths, resize_len=512, crop_height=256, crop_width=256, flag=True):
    if isinstance(paths, str):
        paths = [paths]

    images = []
    for path in paths:
        image = get_image(path, height=crop_height, width=crop_width, set_mode='L')

        if flag:
            image = np.stack(image, axis=0)
            image = np.stack((image, image, image), axis=-1)
        else:
            image = np.stack(image, axis=0)
            image = image.reshape([crop_height, crop_width, 1])
        images.append(image)
    images = np.stack(images, axis=-1)
    return images


def get_train_images_rgb(paths, crop_height=256, crop_width=256, flag=False):
    if isinstance(paths, str):
        paths = [paths]

    images = []
    for path in paths:
        image = get_image(path, height=crop_height, width=crop_width, set_mode='RGB')
        image = np.stack(image, axis=0)
        images.append(image)
    images = np.stack(images, axis=-1)
    return images


def get_test_image_rgb(path, resize_len=512, crop_height=256, crop_width=256, flag = True):
    # image = imread(path, mode='L')
    image = cv2.imread(path)
    return image

def get_test_image_lab(path, resize_len=512, crop_height=776, crop_width=690, flag = True):
    # image = imread(path, mode='L')
    # image = imread(path, pilmode='RGB')
    src = cv2.imread(path)
    #image = cv2.resize(src, (crop_height, crop_width))
    image = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    
    return image


def get_test_image(path, resize_len=512, crop_height=256, crop_width=256, flag = True):
    image = imread(path, pilmode='L')
    image = tf.expand_dims(image,0).eval()
    image = tf.expand_dims(image,-1).eval()
    # image = imread(path, pilmode='RGB')
    return image

def get_images_test(path, mod_type='L', height=None, width=None):

    image = imread(path, mode=mod_type)
    if height is not None and width is not None:
        image = cv2.resize(image, [width, height], cv2.INTER_NEAREST)

    if mod_type=='L':
        d = image.shape
        image = np.reshape(image, [d[0], d[1], 1])

    return image


def get_images(paths, height=None, width=None):
    if isinstance(paths, str):
        paths = [paths]

    images = []
    for path in paths:
        image = imread(path, pilmode='RGB')

        if height is not None and width is not None:
            image = cv2.resize(image, [height, width], cv2.INTER_NEAREST)

        images.append(image)

    images = np.stack(images, axis=0)
    print('images shape gen:', images.shape)
    return images


def save_images(paths, datas, save_path, prefix=None, suffix=None):
    path1 = 'C:/Users/TG/Desktop/imagefusion_densefuse-master1/imagefusion_densefuse-master/images/MF_images/color/0058_rgb.jpg'
    if isinstance(paths, str):
        paths = [paths]
        
    t1 = len(paths)
    t2 = len(datas)
    assert(len(paths) == len(datas))

    if not exists(save_path):
        mkdir(save_path)

    if prefix is None:
        prefix = ''
    if suffix is None:
        suffix = ''

    for i, path in enumerate(paths):
        data = datas[i]
        # print('data ==>>\n', data)
        if data.shape[2] == 1:
            data = data.reshape([data.shape[0], data.shape[1]])
        # print('data reshape==>>\n', data)
        
        #vis_img = get_test_image_rgb(path1, flag=False)
        #Lab = cv2.cvtColor(vis_img, cv2.COLOR_RGB2LAB)
        #a = Lab[:,:,1]
        #b = Lab[:,:,2]
        
        #lab = np.stack((data, a, b), axis=-1)
        
        name, ext = splitext(path)
        name = name.split(sep)[-1]
        #data = np.clip(data,0, 255)
        #data = np.uint8(data)
        
        path = join(save_path, prefix + suffix + ext)
        print('data path==>>', path)
        print(data.shape)
        #new_im = Image.fromarray(data)
        #new_im.show()

        imsave(path, data)
        
def save_images_rgb(paths, datas, save_path, prefix=None, suffix=None):
    if isinstance(paths, str):
        paths = [paths]

    t1 = len(paths)
    t2 = len(datas)
    assert(len(paths) == len(datas))

    if not exists(save_path):
        mkdir(save_path)

    if prefix is None:
        prefix = ''
    if suffix is None:
        suffix = ''

    for i, path in enumerate(paths):
        data = datas[i]
        # print('data ==>>\n', data)
        if data.shape[2] == 1:
            data = data.reshape([data.shape[0], data.shape[1]])
        # print('data reshape==>>\n', data)
        
        name, ext = splitext(path)
        name = name.split(sep)[-1]
        
        path = join(save_path, prefix + suffix + ext)
        print('data path==>>', path)
        #print(rgb.shape)
    
        #new_im = Image.fromarray(data)
        #new_im.show()
        imsave(path, data)
        
def save_images_lab(l, a, b, save_path, prefix=None, suffix=None):

    #name, ext = splitext(path)
    #name = name.split(sep)[-1]
        
    path = join(save_path, prefix + suffix + '.jpg')
    print('data path==>>', path)
    l= (l-np.min(l))/(np.max(l)-np.min(l))*255
    data = np.stack((l.astype('uint8'), a.astype('uint8'), b.astype('uint8')), axis=-1)
    data = cv2.cvtColor(data, cv2.COLOR_LAB2BGR)
        #print(rgb.shape)
    
        #new_im = Image.fromarray(data)
        #new_im.show()
    '''
        cv2.imshow("a",data.astype("uint8"))
        cv2.imshow("b",l.astype('uint8'))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    '''
    cv2.imwrite(path, data)

def save_images_lab1(paths, datas, save_path, prefix=None, suffix=None):
    if isinstance(paths, str):
        paths = [paths]

    t1 = len(paths)
    t2 = len(datas)
    #assert(len(paths) == len(datas))

    if not exists(save_path):
        mkdir(save_path)

    if prefix is None:
        prefix = ''
    if suffix is None:
        suffix = ''

    for i, path in enumerate(paths):
        data = datas[i]
        print(paths)
        print(save_path)
        print(datas[i].shape)
        # print('data ==>>\n', data)
        if data.shape[2] == 1:
            data = data.reshape([data.shape[0], data.shape[1]])
        # print('data reshape==>>\n', dat
   

        name, ext = splitext(path)
        name = name.split(sep)[-1]
        print(data.shape)
        l,a,b = cv2.split(data)
        
        path = join(save_path, prefix + suffix + ext)
        print('data path==>>', path)
        l= (l-np.min(l))/(np.max(l)-np.min(l))*255
        data = np.stack((l.astype('uint8'), a.astype('uint8'), b.astype('uint8')), axis=-1)
        data = cv2.cvtColor(data, cv2.COLOR_LAB2BGR)
        #print(rgb.shape)
    
        #new_im = Image.fromarray(data)
        #new_im.show()
        '''
        cv2.imshow("a",data.astype("uint8"))
        cv2.imshow("b",l.astype('uint8'))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        cv2.imwrite(path, data)


def get_l2_norm_loss(diffs):
    shape = diffs.get_shape().as_list()
    size = reduce(lambda x, y: x * y, shape) ** 2
    sum_of_squared_diffs = tf.reduce_sum(tf.square(diffs))
    return sum_of_squared_diffs / size