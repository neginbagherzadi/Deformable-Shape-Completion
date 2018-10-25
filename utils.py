import numpy as np
import matplotlib.pyplot as plt
#from igraph import *
import glob 
import sys
def load_data():
    path_arr = ["shapes1/*.off", "shapes2/*.off"]
    data_x = []
    data_adj = []
    data_label = []
    
    for item in range(len(path_arr)):
        for mesh in glob.glob(path_arr[item]):            
            adj_input = np.zeros((5000,5000), dtype=float)
            degree_input = np.zeros((5000,5000), dtype=float)
            mesh_target = np.zeros((5000,3), dtype=float)
            with open(mesh, "r") as f:
                array = []
                face = []
                count = 0
                for line in f:
                    count += 1
                    if count >= 3 and count <= 5002: 
                        array.append(line)
                    elif count >= 5003: 
                        face.append(line[2:])
                        face_list = line[2:].split(" ")
                        #put one for evry edge between faces (1,0)=1,(0,1)=1
                        adj_input[int(face_list[0])][int(face_list[1])] = 1.0
                        adj_input[int(face_list[0])][int(face_list[2])] = 1.0
                        adj_input[int(face_list[1])][int(face_list[2])] = 1.0
                        adj_input[int(face_list[1])][int(face_list[0])] = 1.0
                        adj_input[int(face_list[2])][int(face_list[0])] = 1.0
                        adj_input[int(face_list[2])][int(face_list[1])] = 1.0
                
                x,y,z = [],[],[]
                for i in range(5000):
                    xyz_list = array[i].split(" ")            
                    mesh_target[i,0]=float(xyz_list[0])
                    mesh_target[i,1]=float(xyz_list[1])
                    mesh_target[i,2]=float(xyz_list[2])
                data_x.append(mesh_target)    
                
                # print (adj_input)
                degree_column= adj_input.sum(axis=0, dtype='float')
                degree_column=degree_column**(-1/2)
                #print (degree_column)
                np.fill_diagonal(degree_input, degree_column)
                DAD = degree_column*adj_input*degree_column
                #print (DAD)
                data_adj.append(DAD)
                data_label.append(item)
    return data_adj,data_x,data_label        