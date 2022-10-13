# Train the DenseFuse Net

from __future__ import print_function
import cv2
import scipy.io as scio
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import matplotlib.pyplot as plt

from ssim_loss_function import SSIM_LOSS
from densefuse_net import DenseFuseNet
from utils import get_train_images, get_train_images_rgb
import math
from glob import glob

STYLE_LAYERS  = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1')

# TRAINING_IMAGE_SHAPE = (256, 256, 1) # (height, width, color_channels)
# TRAINING_IMAGE_SHAPE_OR = (256, 256, 1) # (height, width, color_channels)
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
HEIGHT = 256
WIDTH = 256
CHANNELS = 1 # gray scale, default

LEARNING_RATE = 1e-4
EPSILON = 1e-5


def train_recons(source1_imgs_path, source2_imgs_path, validation_img1_path, validation_img2_path, validation_target_imgs_path, target_imgs_path, save_path, model_pre_path, ssim_weight, EPOCHES_set, BATCH_SIZE, debug=False, logging_period=1):
#def train_recons(original_imgs_path, validatioin_imgs_path, save_path, model_pre_path, ssim_weight, EPOCHES_set, BATCH_SIZE, debug=False, logging_period=1):
    if debug:
        from datetime import datetime
        start_time = datetime.now()
    EPOCHS = EPOCHES_set
    print("EPOCHES   : ", EPOCHS)
    print("BATCH_SIZE: ", BATCH_SIZE)

    num_val = len(validation_img1_path)
    num_imgs = len(source1_imgs_path)
    # num_imgs = 100
    source1_imgs_path = source1_imgs_path[:num_imgs]
    source2_imgs_path = source2_imgs_path[:num_imgs]
    target_imgs_path = target_imgs_path[:num_imgs]
    
    mod = num_imgs % BATCH_SIZE

    print('Train images number %d.\n' % num_imgs)
    print('Train images samples %s.\n' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        source1_imgs_path = source1_imgs_path[:-mod]
        source2_imgs_path = source2_imgs_path[:-mod]
        target_imgs_path = target_imgs_path[:-mod]

    # get the traing image shape
    INPUT_SHAPE_OR = (BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)
    TARGET_SHAPE_OR = (BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)
    # create the graph
    with tf.Graph().as_default(), tf.Session() as sess:
        source1 = tf.placeholder(tf.float32, shape = INPUT_SHAPE_OR, name='source1')
        source2 = tf.placeholder(tf.float32, shape = INPUT_SHAPE_OR, name='source2')
        target = tf.placeholder(tf.float32, shape = TARGET_SHAPE_OR, name='target')
        source = source1
        
        print('source  :', source.shape)
        print('original:', source1.shape, source2.shape)

        # create the deepfuse net (encoder and decoder) 
    
        dfn = DenseFuseNet(model_pre_path)
        generated_img = dfn.transform_addition(source1, source2)
        
        print('generate:', generated_img.shape)
        print('target:', target.shape)

        ssim_loss_value = SSIM_LOSS(generated_img, target)        
        
        pixel_loss = tf.reduce_sum(tf.square(generated_img - target))      
        pixel_loss = pixel_loss/(BATCH_SIZE*HEIGHT*WIDTH)
        ssim_loss = 1 - ssim_loss_value
        
        loss = ssim_weight*ssim_loss + pixel_loss
        train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        #saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

        # ** Start Training **
        step = 0
        count_loss = 0
        n_batches = int(len(source1_imgs_path) // BATCH_SIZE)
        val_batches = int(len(validation_img1_path) // BATCH_SIZE)

        if debug:
            elapsed_time = datetime.now() - start_time
            print('\nElapsed time for preprocessing before actually train the model: %s' % elapsed_time)
            print('Now begin to train the model...\n')
            start_time = datetime.now()

        Loss_all = [i for i in range(EPOCHS * n_batches)]
        Loss_ssim = [i for i in range(EPOCHS * n_batches)]
        Loss_pixel = [i for i in range(EPOCHS * n_batches)]
        Val_ssim_data = [i for i in range(EPOCHS * n_batches)]
        Val_pixel_data = [i for i in range(EPOCHS * n_batches)]
        for epoch in range(EPOCHS):
            data1 = np.array(source1_imgs_path)
            data2 = np.array(source2_imgs_path)
            data3 = np.array(target_imgs_path)
            
            s = np.arange(data1.shape[0])
            np.random.shuffle(s)
            source1_imgs_path = data1[s]
            source2_imgs_path = data2[s]          
            target_imgs_path = data3[s]
            
            #np.random.shuffle(original_imgs_path)

            for batch in range(n_batches):
                # retrive a batch of content and style images

                source1_path = source1_imgs_path[batch*BATCH_SIZE:(batch*BATCH_SIZE + BATCH_SIZE)]
                source2_path = source2_imgs_path[batch*BATCH_SIZE:(batch*BATCH_SIZE + BATCH_SIZE)]
                target_path = target_imgs_path[batch*BATCH_SIZE:(batch*BATCH_SIZE + BATCH_SIZE)]
                
                ### read gray scale images
                source1_batch = get_train_images(source1_path, crop_height=HEIGHT, crop_width=WIDTH, flag=False)
                source2_batch = get_train_images(source2_path, crop_height=HEIGHT, crop_width=WIDTH, flag=False)
                target_batch = get_train_images(target_path, crop_height=HEIGHT, crop_width=WIDTH, flag=False)
                
                ### read RGB images
                #original_batch = get_train_images_rgb(original_path, crop_height=HEIGHT, crop_width=WIDTH, flag=False)
                #target_batch = get_train_images_rgb(target_path, crop_height=HEIGHT, crop_width=WIDTH, flag=False)
                
                source1_batch = source1_batch.transpose((3, 0, 1, 2))
                source2_batch = source2_batch.transpose((3, 0, 1, 2))
                target_batch = target_batch.transpose((3, 0, 1, 2))
                
                source1_batch = source1_batch.astype('float32')
                source2_batch = source2_batch.astype('float32')
                target_batch = target_batch.astype('float32')
            
                '''
                source1_batch = source1_batch.reshape([256,256,1])
                source2_batch = source2_batch.reshape([256,256,1])
                target_batch = target_batch.reshape([256,256,1])
            
                cv2.imshow('source1',source1_batch)
                cv2.imshow('source2',source2_batch)
                cv2.imshow('target',target_batch)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                source1_batch = source1_batch.reshape([1,256,256,1])
                source2_batch = source2_batch.reshape([1,256,256,1])
                target_batch = target_batch.reshape([1,256,256,1])
                '''
                #이미지 확인용
                
                print('source1_batch shape final:', source1_batch.shape)
                print('source2_batch shape final:', source2_batch.shape)
                
                # run the training step
                sess.run(train_op, feed_dict={target: target_batch, source1: source1_batch, source2: source2_batch})
                step += 1
                #if(step % 6000 == 0):
                    #saver.save(sess, save_path+f'{step}.ckpt', global_step= step)
                if debug:
                    is_last_step = (epoch == EPOCHS - 1) and (batch == n_batches - 1)

                    if is_last_step or step % logging_period == 0:
                        elapsed_time = datetime.now() - start_time
                        _ssim_loss, _loss, _p_loss = sess.run([ssim_loss, loss, pixel_loss], feed_dict={target: target_batch, source1: source1_batch, source2: source2_batch})
                       
                        Loss_all[count_loss] = _loss
                        Loss_ssim[count_loss] = _ssim_loss
                        Loss_pixel[count_loss] = _p_loss
                        print('epoch: %d/%d, step: %d,  total loss: %s, elapsed time: %s' % (epoch, EPOCHS, step, _loss, elapsed_time))
                        print('p_loss: %s, ssim_loss: %s ,w_ssim_loss: %s ' % (_p_loss, _ssim_loss, ssim_weight * _ssim_loss))
                        
                        
                        #print("x: ",sess.run(x),"y: ",sess.run(y))
                        
                    
                        #if math.isnan(_loss) is True :
                          # print("img num: ",step)
                          # break
                        # calculate the accuracy rate for 1000 images, every 100 steps
                        
                        '''
                        val_ssim_acc = 0
                        val_pixel_acc = 0
                        
                        data4 = np.array(validation_img1_path)
                        data5 = np.array(validation_img2_path)
                        data6 = np.array(validation_target_imgs_path)
                        
                        t = np.arange(data4.shape[0])
                        np.random.shuffle(t)
                        validation_img1_path = data4[t]
                        validation_img2_path = data5[t]
                        validation_target_imgs_path = data6[t]
                        
                        val_start_time = datetime.now()
                        for v in range(val_batches):
                            val_original1_path = validation_img1_path[v * BATCH_SIZE:(v * BATCH_SIZE + BATCH_SIZE)]
                            val_original1_batch = get_train_images(val_original1_path, crop_height=HEIGHT, crop_width=WIDTH,flag=False)
                            val_original1_batch = val_original1_batch.reshape([BATCH_SIZE, 256, 256, 1])
                            
                            val_original2_path = validation_img2_path[v * BATCH_SIZE:(v * BATCH_SIZE + BATCH_SIZE)]
                            val_original2_batch = get_train_images(val_original2_path, crop_height=HEIGHT, crop_width=WIDTH,flag=False)
                            val_original2_batch = val_original2_batch.reshape([BATCH_SIZE, 256, 256, 1])
                            
                            val_target_path = validation_target_imgs_path[v * BATCH_SIZE:(v * BATCH_SIZE + BATCH_SIZE)]
                            val_target_batch = get_train_images(val_target_path, crop_height=HEIGHT, crop_width=WIDTH,flag=False)
                            val_target_batch = val_target_batch.reshape([BATCH_SIZE, 256, 256, 1])
                           
                            
                            val_original1_batch = val_original1_batch.reshape([256,256,1])
                            val_original2_batch = val_original2_batch.reshape([256,256,1])
                            val_target_batch = val_target_batch.reshape([256,256,1])
                            
                            cv2.imshow('val_original1', val_original1_batch)
                            cv2.imshow('val_original2',val_original2_batch)
                            cv2.imshow('val_target',val_target_batch)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                            
                            val_original1_batch = val_original1_batch.reshape([1,256,256,1])
                            val_original2_batch = val_original2_batch.reshape([1,256,256,1])
                            val_target_batch = val_target_batch.reshape([1,256,256,1])
                            
                            #이미지 확인용
                            
                            val_ssim, val_pixel = sess.run([ssim_loss, pixel_loss], feed_dict={target: val_target_batch, source1: val_original1_batch, source2: val_original2_batch, })
                            val_ssim_acc = val_ssim_acc + (1-val_ssim)
                            val_pixel_acc = val_pixel_acc + val_pixel
                        Val_ssim_data[count_loss] = val_ssim_acc/val_batches
                        Val_pixel_data[count_loss] = val_pixel_acc / val_batches
                        val_es_time = datetime.now() - val_start_time
                        print('validation value, SSIM: %s, Pixel: %s, elapsed time: %s' % (val_ssim_acc/val_batches, val_pixel_acc / val_batches, val_es_time))
                        print('------------------------------------------------------------------------------')
                        count_loss += 1
                        '''
            #break
        # ** Done Training & Save the model **
        saver.save(sess, save_path)
        
        loss_data = Loss_all[:count_loss]
        scio.savemat('C:/Users/TG/Desktop/imagefusion_densefuse-master1/imagefusion_densefuse-master/models/loss/DeepDenseLossData'+str(ssim_weight)+'.mat',{'loss':loss_data})

        loss_ssim_data = Loss_ssim[:count_loss]
        scio.savemat('C:/Users/TG/Desktop/imagefusion_densefuse-master1/imagefusion_densefuse-master/models/loss/DeepDenseLossSSIMData'+str(ssim_weight)+'.mat', {'loss_ssim': loss_ssim_data})

        loss_pixel_data = Loss_pixel[:count_loss]
        scio.savemat('C:/Users/TG/Desktop/imagefusion_densefuse-master1/imagefusion_densefuse-master/models/loss/DeepDenseLossPixelData.mat'+str(ssim_weight)+'', {'loss_pixel': loss_pixel_data})

        validation_ssim_data = Val_ssim_data[:count_loss]
        scio.savemat('C:/Users/TG/Desktop/imagefusion_densefuse-master1/imagefusion_densefuse-master/models/val/Validation_ssim_Data.mat' + str(ssim_weight) + '', {'val_ssim': validation_ssim_data})

        validation_pixel_data = Val_pixel_data[:count_loss]
        scio.savemat('C:/Users/TG/Desktop/imagefusion_densefuse-master1/imagefusion_densefuse-master/models/val/Validation_pixel_Data.mat' + str(ssim_weight) + '', {'val_pixel': validation_pixel_data})
        
        if debug:
            elapsed_time = datetime.now() - start_time
            print('Done training! Elapsed time: %s' % elapsed_time)
            print('Model is saved to: %s' % save_path)