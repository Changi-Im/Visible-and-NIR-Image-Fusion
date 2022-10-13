# Demo - train the DenseFuse network & use it to generate an image

from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import time
from train_recons_a import train_recons
from generate_t import generate
from utils import list_images
import os
from glob import glob

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# True for training phase
IS_TRAINING = False
# True for video sequences(frames)
IS_VIDEO = False
# True for RGB images
is_RGB = False
# True for One image
IS_ONE = False
# True for Lab
IS_LAB = True
BATCH_SIZE = 1
EPOCHES = 20

SSIM_WEIGHTS = 1000
MODEL_SAVE_PATHS = './models/best/dif_over_47and13_1_50_random_addition_valid.ckpt'
# difference mean over 47 & variance mean over 13 random addition x : y with validation set

# In testing process, 'model_pre_path' is set to None
# The "model_pre_path" in "main.py" is just a pre-train model and not necessary for training and testing. 
# It is set as None when you want to train your own model. 
# If you already train a model, you can set it as your model for initialize weights.
model_pre_path = None
model_save_path = MODEL_SAVE_PATHS

def main():


	if IS_TRAINING:
		
		source1_imgs_path = list_images('./datasets/lum_over_47_var_over_13/ir_crop3')
		source2_imgs_path = list_images('./datasets/lum_over_47_var_over_13/vis_crop3')
		target_imgs_path = list_images('./datasets/lum_over_47_var_over_13/van_crop3')
		validation_img1_path = list_images('./datasets/val/val_test_original1')
		validation_img2_path = list_images('./datasets/val/val_test_original2')
		validation_target_imgs_path = list_images('./datasets/val/val_test_target')
		
		model_save_path = MODEL_SAVE_PATHS
		ssim_weight = SSIM_WEIGHTS
		print('\nBegin to train the network ...\n')
		train_recons(source1_imgs_path, source2_imgs_path, validation_img1_path, validation_img2_path, validation_target_imgs_path, target_imgs_path, model_save_path, model_pre_path, ssim_weight, EPOCHES, BATCH_SIZE, debug=True)		
		print('\nSuccessfully! Done training...\n')
	else:
		if IS_VIDEO:
			ssim_weight = SSIM_WEIGHTS[0]
			model_path = MODEL_SAVE_PATHS[0]

			IR_path = list_images('video/1_IR/')
			VIS_path = list_images('video/1_VIS/')
			output_save_path = 'video/fused'+ str(ssim_weight) +'/'
			generate(IR_path, VIS_path, model_path, model_pre_path,
			         ssim_weight, 0, IS_VIDEO, 'addition', output_path=output_save_path)
		else:
			ssim_weight = SSIM_WEIGHTS
			model_path = MODEL_SAVE_PATHS
			print('\nBegin to generate pictures ...\n')
			path = './inputs/' # you should add images into input folder
			for i in range(0,1):
			 for j in range(0,11):
    				num = f"{i}".zfill(4) # for reading
    				index = i
    
    				# RGB images
    				infrared = path + num + '_nir.jpg'
    				visible = path + num + '_rgb.jpg'
    				fuse = path + num + '_nir.jpg' # it is used only if IS_ONE is true
                    
    				# choose fusion layer, it doesn't matter if IS_LAB is true
    				fusion_type = 'addition'
    				#fusion_type = 'l1'
    				#fusion_type = 'pca'
    				
    				output_save_path = './outputs/'
    				generate(infrared, visible, fuse, model_path, model_pre_path,
    				ssim_weight, index, IS_VIDEO, is_RGB, IS_ONE, IS_LAB, type = fusion_type, output_path = output_save_path)
				        
if __name__ == '__main__':
    main()

