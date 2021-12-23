# -*- coding: utf-8 -*-
""" this file aim to plot the scores (precision,f1,recall) to evaluate a model after training """
#NOTE : this files is only configurated for specific model (at least for image loading), and to replicate the test/train set, use the same seed as for training

#library
#basics
import pandas as pd
import numpy as np
import cv2 as cv
#pytorch
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
#functions file
from functions import *
#counting time
import time

from os import listdir 
from os.path import isfile, join

import scipy.ndimage.filters as filters
from skimage.feature import peak_local_max



#%%
### SEED FOR RANDOM PROCESS
np.random.seed(42)
torch.manual_seed(42)
#%%

#load model

# class
class Pixel14(torch.nn.Module):
    def __init__(self):
        """From: LeCun et al., 1998. Gradient-Based Learning Applied to Document Recognition"""
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 21, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(21, 42, kernel_size=3)
        n = 168
        self.fc1 = torch.nn.Linear(n,n,bias=True)
        self.fc2 = torch.nn.Linear(n,n,bias=True)
        self.fc3 = torch.nn.Linear(n,n,bias=True)
        self.fc4 = torch.nn.Linear(n,n,bias=True)
        self.fc5 = torch.nn.Linear(n,n,bias=True)
        # out layer
        self.fcOut = torch.nn.Linear(n, 1,bias=True)
      
    def forward(self, x):
        relu = torch.nn.functional.relu
        max_pool2d = torch.nn.functional.max_pool2d
        # convolutions
        x = relu(max_pool2d(self.conv1(x),2))
        x = relu(max_pool2d(self.conv2(x),2))
        x = x.view(-1,168)
        # linear layers
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        x = relu(self.fc4(x))
        x = relu(self.fc5(x))
        # test avec 3 layers
        
        # out layer
        x = self.fcOut(x)
        return torch.sigmoid(x)

#path_to_model = 'models/CEP63_only_balanced_twice'
path_to_model = 'models/CP110_only_balanced_twice'

model = torch.load(path_to_model)


#%%
#load data

data = pd.read_csv("data/annotations.csv")

#%%
# load images
# Load all CEP63 (or other folder) images
#path = 'data/RPE1wt_CEP63+CETN2+PCNT_1/CEP63/tif'
# CP110
path = 'data/RPE1wt_CP110+GTU88+PCNT_2/CP110/tif'

images = []
names = [f for f in listdir(path) if isfile(join(path, f))]

for n in names:
    images.append( cv.imread( path + '/' + n , cv.IMREAD_UNCHANGED) )

names = [n[:-4] for n in names]

print( 'number of images ' + str(len(images)) )

#%%



#%%

# load train images

for i in indexes_spl[:spl_nb]:
    spl_images.append(images[i])
    spl_names.append(names[i])
    
#%%
    
# load test images

for i in indexes_spl[spl_nb:]:
    spl_images.append(images[i])
    spl_names.append(names[i])

#%%
    
# load all images images

for i in indexes_spl[:spl_nb]:
    spl_images.append(images[i])
    spl_names.append(names[i])
#%%
#normalize images
images_n = normalizeImages( spl_images )


#%%

#generate prediction map for each images
list_predictions = generatePredictions( images_n, model, 0.005 )


#%%

#Apply different threshold on the map and then store p f r

min_ = 0
max_ = 1
step = 0.05

# vector of threshold to test
vec_threshold = np.arange( min_+step,max_-step,step )

#list to store scores
mean_p = []
mean_f = []
mean_r = []

# iterate on each threshold
for t in vec_threshold:
    #list to store the scores for each images
    p_s=[]
    f_s=[]
    r_s=[]
    
    #iterate on predictions, filter them using the threshold t, and compute the score by comparison to true values
    for i,p in enumerate(list_predictions):
        #trashy be needed to skip empty image
        if(i==14): continue
        
        df = pd.DataFrame( columns = [ 'image_name' , 'x','y','score'  ] )
        #get the punctual maxima of predictions
        peaks = peak_local_max(  p *(p>t ) , min_distance=1 )
        pred_max = np.zeros((2048,2048))
        for p in peaks:
            pred_max[p[0],p[1]] = 1
                
        pred_max = (pred_max >0.)*1.
            
        #get each contours detectect as centrioles
        contours, _ = cv.findContours((pred_max>0.).astype(np.uint8),cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)
        
        
        for j in range(0,len(contours)):
            pt = (contours[j].mean(axis=0)[0]).astype(np.int)
                
            x,y,w,h = cv.boundingRect(contours[j])
                
            s_row = pd.Series([ spl_names[i] , pt[0] , pt[1] , 1. ], index=df.columns)
        
            df = df.append(s_row,ignore_index=True)
        
        p_, f_, r_ = 0,0,0
        if( len(df)==0 ):
            if( len( data[ data['image_name']==spl_names[i] ] ) == 0 ):
                p_, f_, r_ = 1,1,1
            else:p_, f_, r_ = 0,0,1
                
        else:
            p_, f_, r_ = ComputePrecision( df, data[ data['image_name'] == spl_names[i] ])

        p_s.append(p_)
        f_s.append(f_)
        r_s.append(r_)
    # comput ethe mean score acrros each images and store them
    mean_p.append(np.mean(p_s))
    mean_f.append(np.mean(f_s))
    mean_r.append(np.mean(r_s))



#%%

# plot the results

plt.figure(figsize=(20,10))
plt.plot( vec_threshold , mean_p , 'b' , label = 'precision' )
plt.plot( vec_threshold , mean_f , 'g' , label = 'f1' )
plt.plot( vec_threshold , mean_r , 'r' , label = 'recall' )

plt.legend()
#                 name of the stain
plt.title( 'Specific (CP110) Balanced Model Performance depending on prediction filter threshold' )

plt.xticks(vec_threshold)

plt.xlabel('Prediction Threshold value')
plt.ylabel('Scores')









