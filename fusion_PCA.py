import numpy as np
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

def sortMatrix(img, row, col):
    sMat = np.reshape(img, col*row)
    return np.sort(sMat)

#array sort in color image
def sortMatrix_color(img, row, col):
    sMat = np.reshape(img, col*row*3)
    return np.sort(sMat)

def PCA(img1, img2):
    row, col = img1.shape[:2]
    ###when input images are color, change this function###
#    img1_sort = sortMatrix(img1, row, col)
#    img2_sort = sortMatrix(img2, row, col)
    
    img1_sort = np.reshape(img1, col*row)
    img2_sort = np.reshape(img2, col*row)
    #######################################################
    covariance = np.cov(img1_sort, img2_sort)
#    covariance = np.cov(img1, img2)
#    print ('covariance : ', covariance)
    eig = np.linalg.eig(covariance)
    eig_val = eig[0]
    ###revise absolute 1118######
    eig_vec = np.absolute(eig[1])
    #############################
    # print('eig_val : ', eig_val)
    # print('eig_vec : ' , eig_vec)
    # print('eig_vec[0] : ',eig_vec[:,0])
#    d = np.size(d, axis =1)
#    print ('eigenvalue, eigenvector', eig_val, eig_vec)
    
    if eig_val[0] >= eig_val[1]:
        pca = eig_vec[:,0]/sum(eig_vec[:,0])
        # print('pca : ', pca)
    else:
        pca = eig_vec[:,1]/sum(eig_vec[:,1])
        # print('pca : ', pca)
#    result = pca[0]*img1 + pca[1]*img2*0.3
    
    pca_w_1 = pca[0]
    pca_w_2 = pca[1]
            
    return pca_w_1, pca_w_2

def PCA_strategy(enc1, enc2):
    result = []
    dimension = enc1.shape    
    
    for i in range(dimension[3]):
        temp_enc1 = enc1[0,:,:,i]
        temp_enc2 = enc2[0,:,:,i]
        
        img1 = np.reshape(temp_enc1, (dimension[1], dimension[2]))
        img2 = np.reshape(temp_enc2, (dimension[1], dimension[2]))
        
        '''
        cv2.imshow("a",img1.astype('uint8')*10)
        cv2.imshow("b",img2.astype('uint8')*10)
        print("shape",img1.shape)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        
        pca_w_1, pca_w_2 = PCA(img1, img2)
        array_result = pca_w_1*img1 + pca_w_2*img2
        result.append(array_result)
        
    result = np.array(result)
    result = np.stack(result, axis=-1)
    
    result_tf = np.reshape(result, (dimension[0], dimension[1], dimension[2], dimension[3]))
    print(result_tf.shape)

    return result_tf

def PCA_strategy1(source_en_a, source_en_b):
    result = []
    narry_a = source_en_a
    narry_b = source_en_b

    dimension = source_en_a.shape

    # caculate L1-norm
    temp_abs_a = tf.abs(narry_a)
    temp_abs_b = tf.abs(narry_b)
    _l1_a = tf.reduce_sum(temp_abs_a,3)
    _l1_b = tf.reduce_sum(temp_abs_b,3)

    _l1_a = tf.reduce_sum(_l1_a, 0)
    _l1_b = tf.reduce_sum(_l1_b, 0)
    l1_a = _l1_a.eval()
    l1_b = _l1_b.eval()
    
    pca_w_1, pca_w_2 = PCA(l1_a, l1_b)
    
    for i in range(dimension[3]):
        temp_matrix = pca_w_1*narry_a[0,:,:,i] + pca_w_2*narry_b[0,:,:,i]
        result.append(temp_matrix)
        
    result = np.stack(result, axis=-1)

    resule_tf = np.reshape(result, (dimension[0], dimension[1], dimension[2], dimension[3]))

    return resule_tf

    


