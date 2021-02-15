# Deformable-Shape-Completion
  The problem of shape completion is becoming increasingly important in computer vision and computer graphics. Several methods have been proposed for 3D shape completion problems based on convolutional neural networks. These methods are accomplished for rigid objects. However, this work provides a learning-based architecture for the completion of partial non-rigid deformation 3D shapes. Here I implemented the shape reconstruction of an object using Variational autoencoder and graph convolutional neural network. 


  The goal of the method used in this study is to take a partial 3D scan of an object as input and complete the 3D shape as output. For this study, I used 3D human body shapes from the DFAUST dataset [1] includes 3D meshes of full-body shapes used as ground truth (see Figure 1), then partial shapes are prepared inference input to the network. For this purpose, the MeshLab [2] software (open-source mesh processing tool) is used for subsampling data because the scan images are captured at a high frame rate, for solving this problem data subsample from N= 12500 to N=5000. Data are augmented from 70 to 7000 by adding normal distribution noise to all vertices.
  
  ![3](https://user-images.githubusercontent.com/34914017/107974700-069c9700-6fc8-11eb-8fb6-e8c250141dd6.jpg)
  
  
[1] Bogo F, Romero J, Loper M, Black MJ. FAUST: Dataset and evaluation for 3D mesh registration. InProceedings
of the IEEE Conference on Computer Vision and Pattern Recognition 2014 (pp. 3794-3801).

[2] Cignoni, Paolo, et al.” Meshlab: an open-source mesh processing tool.” Eurographics Italian Chapter Conference.
Vol. 2008. 2008S.
  

