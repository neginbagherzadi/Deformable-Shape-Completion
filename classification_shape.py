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

data_adj,data_x,data_label = utils.load_data()
valid_x, valid_adj, valid_label = [], [], []


for i in range(3): 
    valid_x.append(data_x.pop(len(data_x)-1)) 
    valid_x.append(data_x.pop(0)) 
    valid_adj.append(data_adj.pop(len(data_adj)-1)) 
    valid_adj.append(data_adj.pop(0)) 
    valid_label.append(data_label.pop(len(data_label)-1)) 
    valid_label.append(data_label.pop(0)) 



restore_flag = True
to_test = True
N = 5000
batch_size=2
directory = "test_classification_1/"
if not os.path.exists(directory):
    os.makedirs(directory)
    
    
learning_rate = 0.001
training_epochs = 120000
display_step = 10


random_seed = 0
np.random.seed(random_seed)


def plot_loss(loss_arr):
    output = open(directory + "cost.py" , 'w') 
    output.write( "import matplotlib.pyplot as plt" + "\r\n" )
    output.write( "loss = []" + "\r\n" )
    output.write( "steps  = []" + "\r\n" ) 
    for i in range(len(loss_arr)):
        output.write( "steps.append("+ str(i) +")" + "\r\n" ) 
    for i in range(len(loss_arr)): 
        output.write( "loss.append("+ str(loss_arr[i]) +")" + "\r\n" ) 
    output.write( "plt.plot( steps , loss,     color ='g', lw=1 )   " + "\r\n" )
    output.write( "plt.xlabel('Steps', fontsize=14)                   " + "\r\n" )
    output.write( "plt.ylabel('Cost',  fontsize=14)                   " + "\r\n" ) 
    output.write( "plt.show()                                         " + "\r\n" )  
    output.close()
    print ("cost.py file is created!")

# tf Graph input
x  = tf.placeholder(name='x',  dtype=tf.float32, shape=[None,5000,3])
adj  = tf.placeholder(name='adj', dtype=tf.float32, shape=[None,5000,5000])
gt = tf.placeholder(name='gt', dtype=tf.float32, shape=[None,2]) 


# Store layers weight & bias
weights = {
    'h1':  tf.Variable(tf.truncated_normal([5000*3,  5000 ],stddev=0.1, seed=random_seed)),
    'h2':  tf.Variable(tf.truncated_normal([5000,  5000 ],stddev=0.1, seed=random_seed)),
    'h3':  tf.Variable(tf.truncated_normal([5000,  5000],stddev=0.1, seed=random_seed)),
    'out': tf.Variable(tf.truncated_normal([5000,  2  ],stddev=0.1, seed=random_seed))
}
biases = {
    'b1':  tf.Variable(tf.truncated_normal([ 5000 ],stddev=0.1, seed=random_seed)),
    'b2':  tf.Variable(tf.truncated_normal([ 5000 ],stddev=0.1, seed=random_seed)),
    'b3':  tf.Variable(tf.truncated_normal([ 5000],stddev=0.1, seed=random_seed)),
    'out': tf.Variable(tf.truncated_normal([ 2  ],stddev=0.1, seed=random_seed))
}


# Create model
def multilayer_perceptron(x):
    mat1 = tf.matmul(adj, x)
    mat1 = tf.reshape(mat1, [-1, 5000*3])
    layer_1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(mat1, weights['h1']), biases['b1'])), 0.5)
    layer_1 = tf.reshape(layer_1, [-1, 5000,1])
    
    mat2 = tf.matmul(adj,layer_1)
    mat2 = tf.reshape(mat2, [-1, 5000])
    layer_2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(mat2,weights['h2']), biases['b2'])), 0.5)
    layer_2 = tf.reshape(layer_2, [-1, 5000,1])
    
    mat3 = tf.matmul(adj,layer_2)
    mat3 = tf.reshape(mat3, [-1, 5000])
    layer_3 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(mat3,weights['h3']), biases['b3'])), 0.5)
    layer_3 = tf.reshape(layer_3,[-1,5000])
    x_hat = tf.add(tf.matmul(layer_3,weights['out']), biases['out'])
    return x_hat

# Construct model
logits = multilayer_perceptron(x) # logits is scores from the network

# Define loss and optimizer    
#loss_op = tf.losses.mean_squared_error(gt, logits)#, weight_matrix) 
loss_op = tf.nn.softmax_cross_entropy_with_logits(_sentinel=None,labels=gt,logits=logits,dim=-1,name=None)
loss_op = tf.reduce_mean(loss_op)  

for weight in weights:
    loss_op += tf.nn.l2_loss(weights[weight]) * 0.005
    
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op) 
 
# Initializing the variables
init = tf.global_variables_initializer()
loss_arr=[]
with tf.Session() as sess:
    sess.run(init)
   
    # Training cycle
    for iter in range(training_epochs):
        avg_cost = 0.
        x_arr = []
        adj_arr = []
        gt_arr = []
        for batch_num in range(batch_size):
            rand = random.randint(0,len(data_x)-1)
            x_arr.append(data_x[rand])
            adj_arr.append(data_adj[rand])
            if data_label[rand] == 0: 
                gt_arr.append([1, 0])
            else:
                gt_arr.append([0, 1])
            
        
        x_arr = np.reshape(x_arr, (-1, 5000, 3))
        adj_arr = np.reshape(adj_arr, (-1, 5000,5000))
        gt_arr = np.reshape(gt_arr, (-1, 2))
        _, c, scores = sess.run([train_op, loss_op, logits], feed_dict={x: x_arr, gt: gt_arr, adj: adj_arr})
        
        # Display logs per epoch step
        if iter % 1 == 0:
            print('[Iter {}] Loss: {}'.format(iter, c))
            loss_arr.append(c)
            # print "scores:" 
            # print scores
            # print "gt:" 
            # print gt_arr
            
        if iter%25 ==0:
            x_arr = []
            adj_arr = []
            gt_arr = []
            
            for batch_num in range(batch_size):
                rand = random.randint(0,len(valid_x)-1)
                x_arr.append(valid_x[rand])
                adj_arr.append(valid_adj[rand])
                if valid_label[rand] == 0: 
                    gt_arr.append([1, 0])
                else:
                    gt_arr.append([0, 1])
                    
            x_arr = np.reshape(x_arr, (-1, 5000, 3))
            adj_arr = np.reshape(adj_arr, (-1, 5000,5000))
            gt_arr = np.reshape(gt_arr, (-1, 2))
            scores, loss = sess.run([logits, loss_op], feed_dict={x: x_arr, gt: gt_arr, adj: adj_arr})
            print "Valid loss:"
            print loss
            print "scores:" 
            print scores
            print "gt:" 
            print gt_arr

        if iter %500 ==0:
            plot_loss(loss_arr)
            
            
        x_arr = []
        adj_arr = []
        gt_arr = []
    print("Optimization Finished!")