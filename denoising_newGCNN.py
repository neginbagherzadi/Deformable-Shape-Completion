#problem:denoising
# batch=2
# number of image 2
# lr=1e-3
# new graph convolutional(DAD)
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

data_adj,data_x = utils.load_traindata_denoising()
data_label = utils.load_labeldata_denoising()
valid_x, valid_adj = [], []


restore_flag = True
to_test = True
N = 5000
batch_size=2
directory = "test_denoising_newGCNN/"
if not os.path.exists(directory):
    os.makedirs(directory)
    
    
learning_rate = 0.001
training_epochs = 120000
display_step = 10


random_seed = 0
np.random.seed(random_seed)


def count_params():
    "print number of trainable variables"
    size = lambda v: reduce(lambda x, y: x*y, v.get_shape().as_list())
    n = sum(size(v) for v in tf.trainable_variables())
    print "Model size: %dK" % (n/1000,)

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


def create_ply(iter, input_graph):
   
    input_graph = np.reshape(input_graph, (5000, 3))
    with open(directory+"generated_ply_one_model_"+ str(iter) +".ply", "w") as f:
        f.write("ply \n")
        f.write("format ascii 1.0                      \n" )
        f.write("element vertex " + str(input_graph.shape[0])+  "\n" )
        f.write("property float x                      \n" )
        f.write("property float y                      \n" )
        f.write("property float z                      \n" )
        f.write("element face 0                        \n" )
        f.write("property list uchar int vertex_indices\n" )
        f.write("end_header \n") 
        for i in range (input_graph.shape[0]):
            f.write(str(input_graph[i, 0])+" "+str(input_graph[i, 1])+" "+str(input_graph[i, 2])+"\n" )
    print "ply creation is done !"    


#--------------------------------------------------------------------------------------------------------------------------

# tf Graph input
x  = tf.placeholder(name='x',  dtype=tf.float32, shape=[None,5000,3])
adj  = tf.placeholder(name='adj', dtype=tf.float32, shape=[None,5000,5000])
gt = tf.placeholder(name='gt', dtype=tf.float32, shape=[None,5000,3]) 


# Store layers weight & bias
weights = {
    'h1':  tf.Variable(tf.truncated_normal([5000*3,  5000],stddev=0.1, seed=random_seed)),
    'h2':  tf.Variable(tf.truncated_normal([5000  ,   500],stddev=0.1, seed=random_seed)),
    'h3':  tf.Variable(tf.truncated_normal([500   ,   100],stddev=0.1, seed=random_seed)),
    'h4':  tf.Variable(tf.truncated_normal([100   ,   500],stddev=0.1, seed=random_seed)),
    'h5':  tf.Variable(tf.truncated_normal([500   ,  5000],stddev=0.1, seed=random_seed)),
    'h6':  tf.Variable(tf.truncated_normal([5000  , 15000],stddev=0.1, seed=random_seed))
}
biases = {
    'b1':  tf.Variable(tf.truncated_normal([ 5000 ],stddev=0.1, seed=random_seed)),
    'b2':  tf.Variable(tf.truncated_normal([ 500 ],stddev=0.1, seed=random_seed)),
    'b3':  tf.Variable(tf.truncated_normal([ 100 ],stddev=0.1, seed=random_seed)),
    'b4':  tf.Variable(tf.truncated_normal([ 500 ],stddev=0.1, seed=random_seed)),
    'b5':  tf.Variable(tf.truncated_normal([ 5000 ],stddev=0.1, seed=random_seed)),
    'b6':  tf.Variable(tf.truncated_normal([ 15000 ],stddev=0.1, seed=random_seed))
}


# Create model
def denoisingAutoencoder(x):
    mat1 = tf.matmul(adj, x)
    mat1 = tf.reshape(mat1, [-1, 5000*3])
    layer_1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(mat1, weights['h1']), biases['b1'])), 0.5)
    layer_1 = tf.reshape(layer_1, [-1, 5000,1])
    
    mat2 = tf.matmul(adj,layer_1)
    mat2 = tf.reshape(mat2, [-1, 5000])
    layer_2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(mat2,weights['h2']), biases['b2'])), 0.5)
    layer_2 = tf.reshape(layer_2, [-1, 500*1])
    
    layer_3 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer_2,weights['h3']), biases['b3'])), 0.5)
    
    layer_4 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer_3,weights['h4']), biases['b4'])), 0.5)

    layer_5 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer_4,weights['h5']), biases['b5'])), 0.5)
    layer_5 = tf.reshape(layer_5,[-1,5000,1])
    

    mat6 = tf.matmul(adj,layer_5)
    mat6 = tf.reshape(mat6, [-1, 5000])
    layer_6 = tf.add(tf.matmul(mat6,weights['h6']), biases['b6'])
    x_hat = tf.reshape(layer_6,[-1,5000,3])

    return x_hat

# Construct model
logits = denoisingAutoencoder(x) # logits is scores from the network
# Define loss and optimizer    
#loss_op = tf.losses.mean_squared_error(gt, logits)#, weight_matrix) 
recon_loss = tf.losses.mean_squared_error(gt, logits)   
recon_loss = tf.reduce_mean(recon_loss) 


for weight in weights:
    recon_loss += tf.nn.l2_loss(weights[weight]) * 0.005
    
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(recon_loss)
#---------------------------------------------------------------------------------------------------------------------------- 

# Initializing the variables
init = tf.global_variables_initializer()
loss_arr=[]

with tf.Session() as sess:
    sess.run(init)
    
    count_params()
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
            gt_arr.append(data_label[rand])
        
        x_arr = np.reshape(x_arr, (-1, 5000, 3))
        adj_arr = np.reshape(adj_arr, (-1, 5000,5000))
        gt_arr = np.reshape(gt_arr, (-1,  5000, 3))
        
        _, c = sess.run([train_op, recon_loss], feed_dict={x: x_arr, adj: adj_arr, gt: gt_arr})
        
        # Display logs per epoch step
        if iter % 1 == 0:
            print('[Iter {}] Loss: {}'.format(iter, c))
            loss_arr.append(c)



        if iter %500 ==0:
            plot_loss(loss_arr)
            temp = sess.run(logits, feed_dict={x: x_arr,adj: adj_arr})
            temp = np.reshape(temp, (-1, 5000, 3)) 
            create_ply(iter, temp[0])   #visulaize one model
            print("Optimization Finished!")
            