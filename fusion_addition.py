# Additioin
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
def Strategy(content, style):
    #return tf.reduce_sum(content, style)
    x = tf.Variable(1., name='x')
    y = tf.Variable(1., name='y')
    return x*content+y*style
    #return content+style

