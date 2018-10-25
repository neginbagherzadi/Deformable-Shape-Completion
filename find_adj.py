import numpy as np
import matplotlib.pyplot as plt
#from igraph import *
import glob 
import sys

for item in ["shapes1/*.off", "shapes2/*.off"]:
    for mesh in glob.glob(item):            
        adj_input = np.zeros((5000,5000), dtype=float)
        degree_input = np.zeros((5000,5000), dtype=float)
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
        
            # print (adj_input)
            degree_column= adj_input.sum(axis=0, dtype='float')
            # print (degree_column)
            np.fill_diagonal(degree_input, degree_column)
            print (degree_input) 
            # exit(0)
    