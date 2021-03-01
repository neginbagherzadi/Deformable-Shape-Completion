# Deformable-Shape-Completion
  The problem of shape completion is becoming increasingly important in computer vision and computer graphics. Several methods have been proposed for 3D shape completion problems based on convolutional neural networks. These methods are accomplished for rigid objects. However, this work provides a learning-based architecture for the completion of partial non-rigid deformation 3D shapes. Here I implemented the shape reconstruction of an object using Variational autoencoder and graph convolutional neural network. 


  The goal of the method used in this study is to take a partial 3D scan of an object as input and complete the 3D shape as output. For this study, I used 3D human body shapes from the DFAUST dataset [1] includes 3D meshes of full-body shapes used as ground truth (see Figure 1), then partial shapes are prepared inference input to the network. For this purpose, the MeshLab [2] software (open-source mesh processing tool) is used for subsampling data because the scan images are captured at a high frame rate, for solving this problem data subsample from N= 12500 to N=5000. Data are augmented from 70 to 7000 by adding normal distribution noise to all vertices.
  
  ![3](https://user-images.githubusercontent.com/34914017/107974700-069c9700-6fc8-11eb-8fb6-e8c250141dd6.jpg)
  
  
  
  
  
  Reformulating convolutional neural network layers for graph-based models. As we know the convolutional neural network operation to the regular images is done by convolving each of the D input channels with the filter E and summing all the results of this step and add the bias to gaining output feature map. The convolutional operation for irregular graph-based data rearranged the weight matrix to the set of the weight matrices such as M = h × w where w_m ∈ R^(E×D ) each of the weight matrices project input features to the output features. For convolution operation, all the M neighbors project its feature vector with w_m corresponding to its relative position. The activation of pixel i in the output feature map is as follow:
  
                                            y_i = b + ∑(w_m  x_n  (m,i))  
                                                   
                                                   
Where b is the bias term and n(m,i) denotes pixel i’s, m-th relative index of neighbor. in generalization form of the equation, instead of using each neighbor j of a node i with a single weight matrix, the soft-assignment matrices q_m(x_i; x_j) are used across all the M weight matrices, N_iis the set of neighbors of i:

                                           y_i  = b + ∑_(m=1)^((∑_(j∈N_i)(q_m (x_i,x_j ) W_m x_j )/|N_i |))		             
                                                   
for defining the weights, the softmax over a linear transformation is used:

                                          q_m (x_i,x_j)∝exp⁡(u_m^T x_i+v_m^T x_j+c_m)			                               
                                                   
with ∑_(m=1)^M(q_m (x_i,x_j )=1)  and u_m,v_m and c_mare the parameters of the linear transformation. After training the dataset with encoder and decoder pair, the decoder part acts as a generator. At inference, the partial image is given to the network. For this purpose, random latent space with a normal distribution is created. Mean squared error is used for calculating the difference between given partial input and the output which is the result of decoder part. For updating the model Stochastic gradient descent is used.
Denoising Autoencoder: is another method which is used for completion missing parts in 3D meshes. The idea behind denoising autoencoders is in order to force the hidden layer to discover more robust features and prevent it from simply learning the identity, we train the autoencoder to reconstruct the input from a corrupted version of it. We consider a multi-layer Graph Convolutional Network (GCNN) with the following layer-wise propagation rule:

                                         H^((l+1))= σ (D^(-1/2) A D ^(-1/2) H^((l)) W^((l)))			    
                                                      
Here, A = A + I_N is the adjacency matrix of the undirected graph G with added self-connections. I_N is the identity matrix, 

D_ii= ∑_j(A_ij) and W^((l)) is a layer-specific trainable weight matrix. σ (·) denotes an activation function, such as the ReLU (·) = max (0, ·). H^((l))∈ R^(N×D) is the matrix of activations in the l^thlayer; H^((0))= X. 








  
  
[1] Bogo F, Romero J, Loper M, Black MJ. FAUST: Dataset and evaluation for 3D mesh registration. InProceedings
of the IEEE Conference on Computer Vision and Pattern Recognition 2014 (pp. 3794-3801).

[2] Cignoni, Paolo, et al.” Meshlab: an open-source mesh processing tool.” Eurographics Italian Chapter Conference.
Vol. 2008. 2008S
  

