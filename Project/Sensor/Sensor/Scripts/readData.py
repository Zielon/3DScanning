# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

root_dir = "C:/Users/Lukas/Desktop/3DScanning/Project/Sensor/Sensor/Data"

w = 320   
h = 240

files = [f for f in listdir(root_dir) if isfile(join(root_dir, f))]

print (files)

for file in files:

    print(file)

    if "color" in file: 
    
        color_buffer = np.fromfile(join(root_dir, file), np.uint8)

        color_buffer = color_buffer.reshape((h,w,3))
        
        plt.figure()
        plt.imshow(color_buffer[:,:,0:3])
        plt.show()
    
    if "depth" in file: 
    
        depth_buffer = np.fromfile(join(root_dir, file), np.float32)

        depth_buffer = depth_buffer.reshape((h, w))

        plt.figure()
        plt.imshow(depth_buffer)
        plt.show()