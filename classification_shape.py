# batch=2
# number of image 2
# lr=1e-3
# new graph convolutional(DAD)
# classification_shape
from __future__ import division   
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc
from tensorflow.contrib.slim import conv2d as conv
from tensorflow.contrib.slim import conv2d_transpose as deconv
from tensorflow.contrib.slim import batch_norm as batch_norm
from tensorflow.contrib import slim   
import sys, os, glob, random, os.path
import math
import time
import h5py
import time
import utils
from numpy import linalg as LA
from cPickle import load

data_x,data_adj,data_label = utils.load_data()



restore_flag = True
to_test = True
N = 5000
batch_size=2
directory = "test_classification_1/"
if not os.path.exists(directory):
    os.makedirs(directory)
    
    
learning_rate = 1e-3
training_epochs = 120000
display_step = 10


random_seed = 0
np.random.seed(random_seed)


# tf Graph input
x  = tf.placeholder(name='x',  dtype=tf.float32, shape=[None,5000*3])
adj  = tf.placeholder(name='adj', dtype=tf.float32, shape=[None,5000*5000])
gt = tf.placeholder(name='gt', dtype=tf.float32, shape=[None,2]) 


# Store layers weight & bias
weights = {
    'h1':  tf.Variable(tf.truncated_normal([3,   16 ],stddev=0.1, seed=random_seed)),
    'h2':  tf.Variable(tf.truncated_normal([16,  32 ],stddev=0.1, seed=random_seed)),
    'h3':  tf.Variable(tf.truncated_normal([32,  64],stddev=0.1, seed=random_seed)),
    'out': tf.Variable(tf.truncated_normal([64*5000,  2  ],stddev=0.1, seed=random_seed))
}
biases = {
    'b1':  tf.Variable(tf.truncated_normal([ 16 ],stddev=0.1, seed=random_seed)),
    'b2':  tf.Variable(tf.truncated_normal([ 32 ],stddev=0.1, seed=random_seed)),
    'b3':  tf.Variable(tf.truncated_normal([ 64],stddev=0.1, seed=random_seed)),
    'out': tf.Variable(tf.truncated_normal([ 2  ],stddev=0.1, seed=random_seed))
}


# Create model
def multilayer_perceptron(x):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(tf.matmul(adj,x),weights['h1'])), biases['b1'])))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(tf.matmul(adj,layer_1),weights['h2'])), biases['b2'])))
    layer_3 = tf.nn.relu(tf.add(tf.matmul(tf.matmul(adj,layer_2),weights['h3'])), biases['b3'])))
    layer_3 = tf.reshape(layer_3,[-1,64*5000])
    x_hat = tf.add(tf.matmul(layer_3,weights['out']), biases['out'])
    return x_hat

# Construct model
logits = multilayer_perceptron(x) # logits is scores from the network

# Define loss and optimizer    
#loss_op = tf.losses.mean_squared_error(gt, logits)#, weight_matrix) 
loss_op = tf.nn.softmax_cross_entropy_with_logits(_sentinel=None,labels=gt,logits=logits,dim=-1,name=None)
loss_op = tf.reduce_mean(loss_op)  
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op) 
 
# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # load your data, get_input_data() get_data()
    
        
    #gt=gt.astype(np.float32)
    # Training cycle
    for iter in range(training_epochs):
        avg_cost = 0.
        x_arr = []
        adj_arr = []
        gt_arr = []
        for batch_num in range(batch_size):
            rand = random.randint(0,len(data_x))
            x_arr.append(data_x[rand])
            adj_arr.append(data_adj[rand])
            if data_label[rand] == 0:
            
                gt_arr.append([1, 0])
            else:
                gt_arr.append([0, 1])
            
        
        x_arr =   np.reshape(x_arr, (-1, 5000*3))
        adj_arr = np.reshape(adj_arr, (-1, 5000*5000))
        gt_arr =  np.reshape(gt_arr, (-1 , 2))
        _, c = sess.run([train_op, loss_op], feed_dict={x: x_arr, gt: gt_arr, adj: adj_arr})
        
        # Display logs per epoch step
        if iter % display_step == 0:
            print('[Iter {}] Recon: {}'.format(iter,c))
            
        if iter % 1000 == 0:
            temp = sess.run(logits, feed_dict={x: x1})
            temp = np.reshape(temp, (1, 5000, 3)) 
            create_ply(iter, temp ,x1)
            
        x_arr = []
        adj_arr = []
        gt_arr = []
    print("Optimization Finished!")