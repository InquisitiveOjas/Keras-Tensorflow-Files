# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:42:25 2019

@author: ojasr
"""

import h5py
import numpy as np
'''
def getWeightsForLayer(layername , filename):
    with h5py.File(filename , mode = 'r') as f:
        for key in f:
            print (key , f[key])
            o = f[key]
            for key1 in o:
                print (key1,o[key1])
                r = o[key1]
                for key2 in r:
                    print(key2,r[key2])

                    
getWeightsForLayer("conv_2" , "Downloads/Alexnet_May_7.h5")'''

def isGroup(obj):
    if isinstance (obj,h5py.Group):
        return True
    
    return False

def isDataset(obj):
    if isinstance (obj,h5py.Dataset):
        return True
    
    return False
def getDatasetFromGroup (datasets , obj):
    
    if isGroup(obj):
        for key in obj:
            x = obj[key]
            getDatasetFromGroup(datasets , x)
    else:
        datasets.append(obj)
                
def getWeightsForLayer(layerName , fileName):
    weights = []
    
    with h5py.File(fileName , mode = 'r') as f:
        for key in f:
            if layerName in key:
                obj = f[key]
                datasets = []
                getDatasetFromGroup(datasets , obj)
        
                for dataset in datasets:
                    w = np.array(dataset)
                    weights.append(w)
                    
    return weights  

weights = getWeightsForLayer("dense_2" , "Downloads/AlexNet_Sept_19.h5")
print (weights)         
                
                
