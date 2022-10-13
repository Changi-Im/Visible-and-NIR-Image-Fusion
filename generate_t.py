 # Use a trained DenseFuse Net to generate fused images
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import cv2
import time
from datetime import datetime

from fusion_l1norm import L1_norm
from densefuse_net import DenseFuseNet
from fusion_PCA import PCA_strategy, PCA_strategy1
from utils import get_test_image_lab, save_images, get_train_images, get_test_image_rgb, get_test_image, save_images_rgb, save_images_lab


def generate(infrared_path, visible_path, fused_path, model_path, model_pre_path, ssim_weight, index, IS_VIDEO, IS_RGB, IS_ONE, IS_LAB, type='addition', output_path=None):

	if IS_VIDEO:
		print('video_addition')
		_handler_video(infrared_path, visible_path, model_path, model_pre_path, ssim_weight, output_path=output_path)
	else:
		if IS_RGB:
			print('RGB - addition')
			_handler_rgb(infrared_path, visible_path, model_path, model_pre_path, ssim_weight, index,
			         output_path=output_path)

			print('RGB - l1')
			_handler_rgb_l1(infrared_path, visible_path, model_path, model_pre_path, ssim_weight, index,
			             output_path=output_path)
		elif IS_ONE:
			print('one')
			_handler_one(fused_path, model_path, model_pre_path, ssim_weight, index, output_path=output_path)
            
		elif IS_LAB:
			print('lab - addition')
			_handler_lab(infrared_path, visible_path,  model_path, model_pre_path, ssim_weight, index, output_path=output_path)
					
			#print('lab - l1')
			#_handler_lab_l1(infrared_path, visible_path, model_path, model_pre_path, ssim_weight, index, output_path=output_path)

			#print('lab - pca')
			#_handler_lab_pca(infrared_path, visible_path, model_path, model_pre_path, ssim_weight, index, output_path=output_path)

			#print('lab - pca1')
			#_handler_lab_pca1(infrared_path, visible_path, model_path, model_pre_path, ssim_weight, index, output_path=output_path)
            
			#print('lab - one')
			#_handler_lab1(infrared_path, visible_path, model_path, model_pre_path, ssim_weight, index, output_path=output_path)
		else:
			if type == 'addition':
				print('addition')
				_handler(infrared_path, visible_path, model_path, model_pre_path, ssim_weight, index, output_path=output_path)
			elif type == 'l1':
				print('l1')
				_handler_l1(infrared_path, visible_path, fused_path, model_path, model_pre_path, ssim_weight, index, output_path=output_path)
			elif type == 'pca':
				print('pca')
				_handler_pca(infrared_path, visible_path, model_path, model_pre_path, ssim_weight, index, output_path=output_path)    

def _handler(ir_path, vis_path, model_path, model_pre_path, ssim_weight, index, output_path=None):
	ir_img = get_test_image(ir_path, flag=False)
	vis_img = get_test_image(vis_path, flag=False)
    
	# ir_img = get_train_images_rgb(ir_path, flag=False)
	# vis_img = get_train_images_rgb(vis_path, flag=False)


	print('img shape final:', ir_img.shape)

	with tf.Graph().as_default(), tf.Session() as sess:
		infrared_field = tf.placeholder(
			tf.float32, shape=ir_img.shape, name='content')
		visible_field = tf.placeholder(
			tf.float32, shape=ir_img.shape, name='style')
		

		dfn = DenseFuseNet(model_pre_path)

		output_image = dfn.transform_addition(infrared_field, visible_field)
		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)

		output = sess.run(output_image, feed_dict={infrared_field: ir_img, visible_field: vis_img})

		save_images(ir_path, output, output_path,
		            prefix='fused' + str(index), suffix='_densefuse_addition_'+str(ssim_weight))


def _handler_l1(ir_path, vis_path, fs_path, model_path, model_pre_path, ssim_weight, index, output_path=None):
	ir_img = get_test_image(ir_path, flag=False)
	vis_img = get_test_image(vis_path, flag=False)
	fs_img = get_test_image(fs_path, flag=False)

	print('img shape final:', ir_img.shape)

	with tf.Graph().as_default(), tf.Session() as sess:

		# build the dataflow graph
		infrared_field = tf.placeholder(
			tf.float32, shape=ir_img.shape, name='content')
		visible_field = tf.placeholder(
			tf.float32, shape=ir_img.shape, name='style')
		fused_field = tf.placeholder(
			tf.float32, shape=fs_img.shape, name='detail')
        
		dfn = DenseFuseNet(model_pre_path)

		enc_ir = dfn.transform_encoder(infrared_field)
		enc_vis = dfn.transform_encoder(visible_field)
		enc_fs = dfn.transform_encoder(fused_field)
        
		target = tf.placeholder(
		    tf.float32, shape=enc_ir.shape, name='target')

		output_image = dfn.transform_decoder(target)

		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)

		enc_ir_temp, enc_vis_temp, enc_fs_temp = sess.run([enc_ir, enc_vis, enc_fs], feed_dict={infrared_field: ir_img, visible_field: vis_img, fused_field: fs_img})
		feature = L1_norm(enc_ir_temp, enc_vis_temp, enc_fs_temp)

		output = sess.run(output_image, feed_dict={target: feature})
		save_images(ir_path, output, output_path,
		            prefix='fused' + str(index), suffix='_densefuse_l1norm_'+str(ssim_weight))
        
def _handler_pca(ir_path, vis_path, model_path, model_pre_path, ssim_weight, index, output_path=None):
	ir_img = get_test_image(ir_path, flag=False)
	vis_img = get_test_image(vis_path, flag=False)
	dimension = ir_img.shape

	#ir_img = ir_img.reshape([1, dimension[0], dimension[1], dimension[2]])
	#vis_img = vis_img.reshape([1, dimension[0], dimension[1], dimension[2]])

	# ir_img = np.transpose(ir_img, (0, 2, 1, 3))
	# vis_img = np.transpose(vis_img, (0, 2, 1, 3))

	print('img shape final:', ir_img.shape)

	with tf.Graph().as_default(), tf.Session() as sess:

		# build the dataflow graph
		infrared_field = tf.placeholder(
			tf.float32, shape=ir_img.shape, name='content')
		visible_field = tf.placeholder(
			tf.float32, shape=ir_img.shape, name='style')

		dfn = DenseFuseNet(model_pre_path)

		enc_ir = dfn.transform_encoder(infrared_field)
		enc_vis = dfn.transform_encoder(visible_field)

		target = tf.placeholder(
		    tf.float32, shape=enc_ir.shape, name='target')

		output_image = dfn.transform_decoder(target)

		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)

		enc_ir_temp, enc_vis_temp = sess.run([enc_ir, enc_vis], feed_dict={infrared_field: ir_img, visible_field: vis_img})
		feature = PCA_strategy(enc_ir_temp, enc_vis_temp)

		output = sess.run(output_image, feed_dict={target: feature})
		save_images(ir_path, output, output_path,
		            prefix='fused' + str(index), suffix='_densefuse_PCA_'+str(ssim_weight))

def _handler_one(fused_path, model_path, model_pre_path, ssim_weight, index, output_path=None):
    start = time.time()
    fs_img = get_test_image(fused_path, flag=False)
    dimension = fs_img.shape
    
    print('img shape final:', fs_img.shape)

    with tf.Graph().as_default(), tf.Session() as sess:

		# build the dataflow graph
        fused_field = tf.placeholder(
			tf.float32, shape=fs_img.shape, name='one')
        dfn = DenseFuseNet(model_pre_path)
        
        enc_fs = dfn.transform_encoder(fused_field)
		
        target = tf.placeholder(
		    tf.float32, shape=enc_fs.shape, name='target')

        output_image = dfn.transform_decoder(target)

		# restore the trained model and run the style transferring
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        enc_fs_temp = sess.run([enc_fs], feed_dict={fused_field: fs_img})
        enc_result = np.reshape(enc_fs_temp, (dimension[0], dimension[1], dimension[2], 64))

		#print(enc_fs_temp.shape)
        output = sess.run(output_image, feed_dict={target: enc_result})
        save_images(fused_path, output, output_path,
		            prefix='fused' + str(index), suffix='_densefuse_one_'+str(ssim_weight))
        t = time.time()-start
        print("a",t)
        
def _handler_video(ir_path, vis_path, model_path, model_pre_path, ssim_weight, output_path=None):
	infrared = ir_path[0]
	img = get_train_images(infrared, flag=False)
	img = img.reshape([1, img.shape[0], img.shape[1], img.shape[2]])
	img = np.transpose(img, (0, 2, 1, 3))
	print('img shape final:', img.shape)
	num_imgs = len(ir_path)

	with tf.Graph().as_default(), tf.Session() as sess:
		# build the dataflow graph
		infrared_field = tf.placeholder(
			tf.float32, shape=img.shape, name='content')
		visible_field = tf.placeholder(
			tf.float32, shape=img.shape, name='style')

		dfn = DenseFuseNet(model_pre_path)

		output_image = dfn.transform_addition(infrared_field, visible_field)

		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)

		##################GET IMAGES###################################################################################
		start_time = datetime.now()
		for i in range(num_imgs):
			print('image number:', i)
			infrared = ir_path[i]
			visible = vis_path[i]

			ir_img = get_train_images(infrared, flag=False)
			vis_img = get_train_images(visible, flag=False)
			dimension = ir_img.shape

			ir_img = ir_img.reshape([1, dimension[0], dimension[1], dimension[2]])
			vis_img = vis_img.reshape([1, dimension[0], dimension[1], dimension[2]])

			ir_img = np.transpose(ir_img, (0, 2, 1, 3))
			vis_img = np.transpose(vis_img, (0, 2, 1, 3))

			################FEED########################################
			output = sess.run(output_image, feed_dict={infrared_field: ir_img, visible_field: vis_img})
			save_images(infrared, output, output_path,
			            prefix='fused' + str(i), suffix='_addition_' + str(ssim_weight))
			######################################################################################################
		elapsed_time = datetime.now() - start_time
		print('Dense block video==> elapsed time: %s' % (elapsed_time))


def _handler_rgb(ir_path, vis_path, model_path, model_pre_path, ssim_weight, index, output_path=None):
	# ir_img = get_train_images(ir_path, flag=False)
	# vis_img = get_train_images(vis_path, flag=False)
	ir_img = get_test_image_rgb(ir_path, flag=False)
	vis_img = get_test_image_rgb(vis_path, flag=False)
	dimension = ir_img.shape

	ir_img = ir_img.reshape([1, dimension[0], dimension[1], dimension[2]])
	vis_img = vis_img.reshape([1, dimension[0], dimension[1], dimension[2]])

	#ir_img = np.transpose(ir_img, (0, 2, 1, 3))
	#vis_img = np.transpose(vis_img, (0, 2, 1, 3))

	ir_img1 = ir_img[:, :, :, 0]
	ir_img1 = ir_img1.reshape([1, dimension[0], dimension[1], 1])
	ir_img2 = ir_img[:, :, :, 1]
	ir_img2 = ir_img2.reshape([1, dimension[0], dimension[1], 1])
	ir_img3 = ir_img[:, :, :, 2]
	ir_img3 = ir_img3.reshape([1, dimension[0], dimension[1], 1])
	ir_l = ir_img1*0.1 + ir_img2*0.6 + ir_img3*0.3 
    
	vis_img1 = vis_img[:, :, :, 0]
	vis_img1 = vis_img1.reshape([1, dimension[0], dimension[1], 1])
	vis_img2 = vis_img[:, :, :, 1]
	vis_img2 = vis_img2.reshape([1, dimension[0], dimension[1], 1])
	vis_img3 = vis_img[:, :, :, 2]
	vis_img3 = vis_img3.reshape([1, dimension[0], dimension[1], 1])
	vis_l = vis_img1*0.1 + vis_img2*0.6 + vis_img3*0.3
    

	print('img shape final:', ir_img1.shape)

	with tf.Graph().as_default(), tf.Session() as sess:
		infrared_field = tf.placeholder(
			tf.float32, shape=ir_img1.shape, name='content')
		visible_field = tf.placeholder(
			tf.float32, shape=ir_img1.shape, name='style')

		dfn = DenseFuseNet(model_pre_path)

		output_image = dfn.transform_addition(infrared_field, visible_field)
		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)

		output1 = sess.run(output_image, feed_dict={infrared_field: ir_img1, visible_field: vis_img1})
		output2 = sess.run(output_image, feed_dict={infrared_field: ir_img2, visible_field: vis_img2})
		output3 = sess.run(output_image, feed_dict={infrared_field: ir_img3, visible_field: vis_img3})

		output1 = output1.reshape([1, dimension[0], dimension[1]])
		output2 = output2.reshape([1, dimension[0], dimension[1]])
		output3 = output3.reshape([1, dimension[0], dimension[1]])

		output = np.stack((output1, output2, output3), axis=-1)
		#output = np.transpose(output, (0, 2, 1, 3))
		save_images_lab(ir_path, output, output_path,
		            prefix='fused' + str(index), suffix='_densefuse_addition_'+str(ssim_weight))


def _handler_rgb_l1(ir_path, vis_path, model_path, model_pre_path, ssim_weight, index, output_path=None):
	# ir_img = get_train_images(ir_path, flag=False)
	# vis_img = get_train_images(vis_path, flag=False)
	ir_img = get_test_image_rgb(ir_path, flag=False)
	vis_img = get_test_image_rgb(vis_path, flag=False)
	dimension = ir_img.shape

	ir_img = ir_img.reshape([1, dimension[0], dimension[1], dimension[2]])
	vis_img = vis_img.reshape([1, dimension[0], dimension[1], dimension[2]])

	#ir_img = np.transpose(ir_img, (0, 2, 1, 3))
	#vis_img = np.transpose(vis_img, (0, 2, 1, 3))

	ir_img1 = ir_img[:, :, :, 0]
	ir_img1 = ir_img1.reshape([1, dimension[0], dimension[1], 1])
	ir_img2 = ir_img[:, :, :, 1]
	ir_img2 = ir_img2.reshape([1, dimension[0], dimension[1], 1])
	ir_img3 = ir_img[:, :, :, 2]
	ir_img3 = ir_img3.reshape([1, dimension[0], dimension[1], 1])

	vis_img1 = vis_img[:, :, :, 0]
	vis_img1 = vis_img1.reshape([1, dimension[0], dimension[1], 1])
	vis_img2 = vis_img[:, :, :, 1]
	vis_img2 = vis_img2.reshape([1, dimension[0], dimension[1], 1])
	vis_img3 = vis_img[:, :, :, 2]
	vis_img3 = vis_img3.reshape([1, dimension[0], dimension[1], 1])

	print('img shape final:', ir_img1.shape)

	with tf.Graph().as_default(), tf.Session() as sess:
		infrared_field = tf.placeholder(
			tf.float32, shape=ir_img1.shape, name='content')
		visible_field = tf.placeholder(
			tf.float32, shape=ir_img1.shape, name='style')

		dfn = DenseFuseNet(model_pre_path)

		enc_ir = dfn.transform_encoder(infrared_field)
		enc_vis = dfn.transform_encoder(visible_field)

		target = tf.placeholder(
			tf.float32, shape=enc_ir.shape, name='target')

		output_image = dfn.transform_decoder(target)

		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)

		enc_ir_temp, enc_vis_temp = sess.run([enc_ir, enc_vis], feed_dict={infrared_field: ir_img1, visible_field: vis_img1})
		feature = L1_norm(enc_ir_temp, enc_vis_temp)
		output1 = sess.run(output_image, feed_dict={target: feature})

		enc_ir_temp, enc_vis_temp = sess.run([enc_ir, enc_vis], feed_dict={infrared_field: ir_img2, visible_field: vis_img2})
		feature = L1_norm(enc_ir_temp, enc_vis_temp)
		output2 = sess.run(output_image, feed_dict={target: feature})

		enc_ir_temp, enc_vis_temp = sess.run([enc_ir, enc_vis], feed_dict={infrared_field: ir_img3, visible_field: vis_img3})
		feature = L1_norm(enc_ir_temp, enc_vis_temp)
		output3 = sess.run(output_image, feed_dict={target: feature})

		output1 = output1.reshape([1, dimension[0], dimension[1]])
		output2 = output2.reshape([1, dimension[0], dimension[1]])
		output3 = output3.reshape([1, dimension[0], dimension[1]])
		print("out",output1.shape)
		output = np.stack((output1, output2, output3), axis=-1)
		print("ff",output.shape)
		#output = np.transpose(output, (0, 2, 1, 3))
		save_images(ir_path, output, output_path,
		            prefix='fused' + str(index), suffix='_densefuse_l1norm_'+str(ssim_weight))
        
def _handler_lab_pca(ir_path, vis_path, model_path, model_pre_path, ssim_weight, index, output_path=None):
	# ir_img = get_train_images(ir_path, flag=False)
	# vis_img = get_train_images(vis_path, flag=False)
	start = time.time()
	ir_img = get_test_image_lab(ir_path, flag=False)
	vis_img = get_test_image_lab(vis_path, flag=False)
	dimension = ir_img.shape
    
	ir_gray,ir_a,ir_b = cv2.split(ir_img)
	vis_gray,a,b = cv2.split(vis_img)  
    
	ir_gray = ir_gray.reshape([1, dimension[0], dimension[1], 1])
	vis_gray = vis_gray.reshape([1, dimension[0], dimension[1], 1])
	

	print('img shape final:', ir_gray.shape)

	with tf.Graph().as_default(), tf.Session() as sess:
		infrared_field = tf.placeholder(
			tf.float32, shape=ir_gray.shape, name='content')
		visible_field = tf.placeholder(
			tf.float32, shape=ir_gray.shape, name='style')

		dfn = DenseFuseNet(model_pre_path)

		enc_ir = dfn.transform_encoder(infrared_field)
		enc_vis = dfn.transform_encoder(visible_field)

		target = tf.placeholder(
			tf.float32, shape=enc_ir.shape, name='target')

		output_image = dfn.transform_decoder(target)

		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)

		enc_ir_temp, enc_vis_temp = sess.run([enc_ir, enc_vis], feed_dict={infrared_field: ir_gray, visible_field: vis_gray})
		feature = PCA_strategy(enc_ir_temp, enc_vis_temp)
		output1 = sess.run(output_image, feed_dict={target: feature})

		L = output1.reshape([1, dimension[0], dimension[1]])
		a = a.reshape([1,dimension[0], dimension[1]])
		b = b.reshape([1,dimension[0], dimension[1]])        
		output = np.stack((L, a, b), axis=-1)
		#output = np.transpose(output, (0, 2, 1, 3))
		save_images_lab(ir_path, output, output_path,
		            prefix='fused' + str(index), suffix='_densefuse_pca_'+str(ssim_weight))
		t= time.time()-start
		print("pca: ",round(t,3)) 
        
def _handler_lab_pca1(ir_path, vis_path, model_path, model_pre_path, ssim_weight, index, output_path=None):
	# ir_img = get_train_images(ir_path, flag=False)
	# vis_img = get_train_images(vis_path, flag=False)
	start = time.time()
	ir_img = get_test_image_lab(ir_path, flag=False)
	vis_img = get_test_image_lab(vis_path, flag=False)
	dimension = ir_img.shape
    
	ir_gray,ir_a,ir_b = cv2.split(ir_img)
	vis_gray,a,b = cv2.split(vis_img)  
    
    
	ir_gray = ir_gray.reshape([1, dimension[0], dimension[1], 1])
	vis_gray = vis_gray.reshape([1, dimension[0], dimension[1], 1])
	

	print('img shape final:', ir_gray.shape)

	with tf.Graph().as_default(), tf.Session() as sess:
		infrared_field = tf.placeholder(
			tf.float32, shape=ir_gray.shape, name='content')
		visible_field = tf.placeholder(
			tf.float32, shape=ir_gray.shape, name='style')

		dfn = DenseFuseNet(model_pre_path)

		enc_ir = dfn.transform_encoder(infrared_field)
		enc_vis = dfn.transform_encoder(visible_field)

		target = tf.placeholder(
			tf.float32, shape=enc_ir.shape, name='target')

		output_image = dfn.transform_decoder(target)

		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)

		enc_ir_temp, enc_vis_temp = sess.run([enc_ir, enc_vis], feed_dict={infrared_field: ir_gray, visible_field: vis_gray})
		feature = PCA_strategy1(enc_ir_temp, enc_vis_temp)
		output1 = sess.run(output_image, feed_dict={target: feature})

		L = output1.reshape([1, dimension[0], dimension[1]])
		a = a.reshape([1,dimension[0], dimension[1]])
		b = b.reshape([1,dimension[0], dimension[1]])        
		output = np.stack((L, a, b), axis=-1)
		#output = np.transpose(output, (0, 2, 1, 3))
		save_images_lab(ir_path, output, output_path,
		            prefix='fused' + str(index), suffix='_densefuse_pca1_'+str(ssim_weight))
		t= time.time()-start
		print("pca1: ",round(t,3)) 
        
def _handler_lab_l1(ir_path, vis_path, model_path, model_pre_path, ssim_weight, index, output_path=None):
	# ir_img = get_train_images(ir_path, flag=False)
	# vis_img = get_train_images(vis_path, flag=False)
	start = time.time()
	ir_img = get_test_image_lab(ir_path, flag=False)
	vis_img = get_test_image_lab(vis_path, flag=False)
	dimension = ir_img.shape
    
	ir_gray,ir_a,ir_b = cv2.split(ir_img)
	vis_gray,a,b = cv2.split(vis_img)   
    
	ir_gray = ir_gray.reshape([1, dimension[0], dimension[1], 1])
	vis_gray = vis_gray.reshape([1, dimension[0], dimension[1], 1])
	
	print('img shape final:', ir_gray.shape)

	with tf.Graph().as_default(), tf.Session() as sess:
		infrared_field = tf.placeholder(
			tf.float32, shape=ir_gray.shape, name='content')
		visible_field = tf.placeholder(
			tf.float32, shape=ir_gray.shape, name='style')

		dfn = DenseFuseNet(model_pre_path)

		enc_ir = dfn.transform_encoder(infrared_field)
		enc_vis = dfn.transform_encoder(visible_field)

		target = tf.placeholder(
			tf.float32, shape=enc_ir.shape, name='target')

		output_image = dfn.transform_decoder(target)

		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)

		enc_ir_temp, enc_vis_temp = sess.run([enc_ir, enc_vis], feed_dict={infrared_field: ir_gray, visible_field: vis_gray})
		feature = L1_norm(enc_ir_temp, enc_vis_temp)
		output1 = sess.run(output_image, feed_dict={target: feature})

		L = output1.reshape([1, dimension[0], dimension[1]])
		a = a.reshape([1,dimension[0], dimension[1]])
		b = b.reshape([1,dimension[0], dimension[1]])        
		output = np.stack((L, a, b), axis=-1)
		#output = np.transpose(output, (0, 2, 1, 3))
		save_images_lab(ir_path, output, output_path,
		            prefix='fused' + str(index), suffix='_densefuse_l1_'+str(ssim_weight))
		t= time.time()-start
		print("L1: ",round(t,3)) 
        
def _handler_lab(ir_path, vis_path, model_path, model_pre_path, ssim_weight, index, output_path=None):
	# ir_img = get_train_images(ir_path, flag=False)
	# vis_img = get_train_images(vis_path, flag=False)
	
	ir_img = get_test_image_lab(ir_path, flag=False)
	vis_img = get_test_image_lab(vis_path, flag=False)
	fusion = time.time()
	dimension = ir_img.shape
    
	ir_gray= ir_img[:,:,0]
	vis_gray,a,b = cv2.split(vis_img)
    
	ir_gray = ir_gray.reshape([1, dimension[0], dimension[1], 1])
	vis_gray = vis_gray.reshape([1, dimension[0], dimension[1], 1])
	#print("split:",time.time()-start)
	
	#ir_img = np.transpose(ir_img, (0, 2, 1, 3))
	#vis_img = np.transpose(vis_img, (0, 2, 1, 3))

	#ir_img1 = ir_img[:, :, :, 0]
	#ir_img1 = ir_img1.reshape([1, dimension[0], dimension[1], 1])
    
            
	#vis_img1 = vis_img[:, :, :, 0] 
	#vis_img1 = vis_img1.reshape([1, dimension[0], dimension[1], 1])
	
	print('img shape final:', ir_gray.shape)

	with tf.Graph().as_default(), tf.Session() as sess:
		infrared_field = tf.placeholder(
			tf.float32, shape=ir_gray.shape, name='content')
		visible_field = tf.placeholder(
			tf.float32, shape=ir_gray.shape, name='style')

		dfn = DenseFuseNet(model_pre_path)

		output_image = dfn.transform_addition(infrared_field, visible_field)

		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)
		
		start = time.time()
		#output1, feature1, feature2, feature_fused = sess.run(output_image, feed_dict={infrared_field: ir_gray, visible_field: vis_gray})
		output1 = sess.run(output_image, feed_dict={infrared_field: ir_gray, visible_field: vis_gray})
        
        
		'''feature1 = feature1.reshape([dimension[0], dimension[1]])
		feature2 = feature2.reshape([dimension[0], dimension[1]])
		feature_fused = feature_fused.reshape([dimension[0], dimension[1]])'''
        
        
		L = output1.reshape([dimension[0], dimension[1]])
		L = (L-np.min(L))/(np.max(L)-np.min(L))*255
		#output = cv2.cvtColor(Lab, cv2.COLOR_LAB2RGB) 
		#output = np.transpose(output, (0, 2, 1, 3))
		Lab = cv2.merge((L.astype('uint8'),a.astype('uint8'),b.astype('uint8')))
		output = cv2.cvtColor(Lab,cv2.COLOR_LAB2BGR)
		#print("x:",sess.run(x),"y:", sess.run(y))
		prefix='fused' + str(index)
		suffix='_densefuse_addition_'+str(ssim_weight)
		#path = output_path + prefix + suffix + '.jpg'
		path = output_path + str(index) + "_random.jpg"
        
		'''path1 = output_path + str(index) + "_random1.jpg"
		path2 = output_path + str(index) + "_random2.jpg"
		path3 = output_path + str(index) + "_random3.jpg"'''
		
        #output = 255.0*output*3/np.max(output*)
		cv2.imwrite(path, np.uint8(output))
		print("elapsed time:",round(time.time()-start,4))
		'''cv2.imwrite(path1, np.uint8(feature1)*4)
		cv2.imwrite(path2, np.uint8(feature2)*4)
		cv2.imwrite(path3, np.uint8(feature_fused)*4)'''
        
def _handler_lab1(ir_path, vis_path, model_path, model_pre_path, ssim_weight, index, output_path=None):
	# ir_img = get_train_images(ir_path, flag=False)
	# vis_img = get_train_images(vis_path, flag=False)
	start = time.time()
	ir_img = get_test_image_lab(ir_path, flag=False)
	vis_img = get_test_image_lab(vis_path, flag=False)
	dimension = ir_img.shape
    
	ir_gray,ir_a,ir_b = cv2.split(ir_img)
	vis_gray,a,b = cv2.split(vis_img)  
    
	l = ir_gray/2+vis_gray/2
	l = l.reshape(1, dimension[0], dimension[1], 1)
	
	#ir_img = np.transpose(ir_img, (0, 2, 1, 3))
	#vis_img = np.transpose(vis_img, (0, 2, 1, 3))

	#ir_img1 = ir_img[:, :, :, 0]
	#ir_img1 = ir_img1.reshape([1, dimension[0], dimension[1], 1])
    
            
	#vis_img1 = vis_img[:, :, :, 0] 
	#vis_img1 = vis_img1.reshape([1, dimension[0], dimension[1], 1])

	print('img shape final:', l.shape)

	with tf.Graph().as_default(), tf.Session() as sess:
		# build the dataflow graph
		fused_field = tf.placeholder(
			tf.float32, shape=l.shape, name='one')
		dfn = DenseFuseNet(model_pre_path)
        
		enc_fs = dfn.transform_encoder(fused_field)
		
		target = tf.placeholder(
		    tf.float32, shape=enc_fs.shape, name='target')

		output_image = dfn.transform_decoder(target)

		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)

		enc_fs_temp = sess.run([enc_fs], feed_dict={fused_field: l})
		enc_result = np.reshape(enc_fs_temp, (1, dimension[0], dimension[1], 64))

		#print(enc_fs_temp.shape)
		output = sess.run(output_image, feed_dict={target: enc_result})
        

		L = output.reshape([1,dimension[0], dimension[1]])
		a = a.reshape([1,dimension[0], dimension[1]])
		b = b.reshape([1,dimension[0], dimension[1]]) 
		
		output = np.stack((L, a, b), axis=-1)
		#output = cv2.cvtColor(Lab, cv2.COLOR_LAB2RGB) 
		#output = np.transpose(output, (0, 2, 1, 3))
		save_images_lab(ir_path, output, output_path,
		            prefix='fused' + str(index), suffix='_densefuse_one_'+str(ssim_weight))
		t= time.time()-start
		print("one: ",round(t,3))