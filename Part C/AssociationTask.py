
#Imports

import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage import measure, color

import torch
from torch import nn
from torch.nn import functional as F

import tensorflow as tf
print(tf.__version__)
layers = tf.keras.layers

from csbdeep.utils import normalize
from stardist.models import StarDist2D 

# prints a list of available models 
StarDist2D.from_pretrained() 


path_ = "data/datasets_full/"

# DAPI is ALWAYS C0

options = [["RPE1wt_CEP152+GTU88+PCNT_1","DAPI","CEP152","GTU88","PCNT"], #Channel 0 : C0,C1,C2,C3
           ["RPE1wt_CEP63+CETN2+PCNT_1","DAPI","CEP63","CETN2","PCNT"], #Channel 1 : C0,C1,C2,C3
           ["RPE1wt_CP110+GTU88+PCNT_2","DAPI","CP110","GTU88","PCNT"]] #Channel 2 : C0,C1,C2,C3

#***************************************************************************************************
#********************************** Utility Functions *************************************************
#***************************************************************************************************


def generatePaths(channel, id_, format_="tif", path=path_):
    """ Get 4 paths and title of sample """
    # generate path strings
    paths = [(path + 
            options[channel][0] + "/" +
            options[channel][C] + "/" +
            format_ + "/" +
            options[channel][0] +
            "_00" + str(id_[0]) + "_00" + str(id_[1]) + "_max_C" + str(C-1) +
            "." + format_)
            for C in range(1,5)]
    
    # generate main_title
    main_title = options[channel][0] +"_00" + str(id_[0]) + "_00" + str(id_[1])
    
    return paths, main_title

""" Get 4 names of subchannels C0,C1,C2,C3 """
def generateNames(channel, id_, format_="tif", path = path_):
    titles = [options[channel][0] +"_00" + str(id_[0]) + "_00" + str(id_[1]) + "_max_C" + str(C-1) for C in range(1,5)] 
    return titles

def count(figure,x,y,value,color):
    position = (x, y)
    cv.putText(
        figure, #numpy array on which text is written
        str(value), #text
        position, #position at which writing has to start
        cv.FONT_HERSHEY_SIMPLEX, #font family
        1, #font size
        (color,0,0), #font color
        3,
        lineType=cv.LINE_AA) #font stroke
    return figure

'''Assign centrioles coordinates to nucleus coordinates'''
def AssingCtr (Ctrs_df, labels, infos, W_size = 6):
    #Ctrs_df : dataframe fo centrioles coordinates
    #labels : labels array representing nucleus prediction
    #points : list of nuclei center coordinates
    #W_size : window size in which to look for proximal nucleus
    
    #size factor of nuclei prediction and base size of 2048
    size_factor = 2048//len(labels[:,0])
    #load x and y centrioles data
    x_, y_ = Ctrs_df.x//size_factor, Ctrs_df.y//size_factor
    #load points of nuclei and probas
    points = infos['points']
    probas = infos['prob']

    #Prepare dict to return, coordinates in original image size
    CtrDict = {label : [] for label in np.unique(labels)}
    CtrDict_B = {label : [] for label in np.unique(labels)}
    #Distances to proximal nucleus list
    Distances = []
    #Distances dictionnary for plotting convinience
    DistDic = {label : [] for label in np.unique(labels)}

    for i, (x, y) in enumerate(zip(x_, y_)):
        for w in range(1,W_size):
            x_min = x - w if x - w>0 else 0
            x_Max = x + w if x + w<len(labels[:,0]) else len(labels[:,0])

            y_min = y - w if y - w>0 else 0
            y_Max = y + w if y + w<len(labels[0,:]) else len(labels[0,:])

            #Check in a square window of size W_size the proximal nucleus
            sub_mask = labels[y_min:y_Max, x_min:x_Max]
            ncl_labels = np.unique(sub_mask)

            '''Still issues with probabilities associated with stardist'''
            #First label detected is within a nucleus
            #if len(ncl_labels) == 1 and ncl_labels[0] != 0 and probas[ncl_labels[0]-1] > 0.5 :
            if len(ncl_labels) == 1 and ncl_labels[0] != 0:
                CtrDict[ncl_labels[0]].append([x, y])
                CtrDict_B[ncl_labels[0]].append([Ctrs_df.x.iloc[i],Ctrs_df.y.iloc[i]])
                Distances.append(0)
                DistDic[ncl_labels[0]].append(0)
                break

            #Centrioles is outside a nucleus
            #if len(ncl_labels) > 1 and ncl_labels[0] == 0 and probas[ncl_labels[1]-1] > 0.5:
            if len(ncl_labels) > 1 and ncl_labels[0] == 0:
                CtrDict[ncl_labels[1]].append([x, y])
                CtrDict_B[ncl_labels[1]].append([Ctrs_df.x.iloc[i],Ctrs_df.y.iloc[i]])
                point = np.where(sub_mask == ncl_labels[1])
                Distances.append(np.linalg.norm([point[1] - w//2, point[0] - w//2]))
                DistDic[ncl_labels[1]].append(np.linalg.norm([point[1] - w//2, point[0] - w//2]))
                break

    #CtrDict : dictionnary of nuclei-centrioles in 256x256
    # Distances : all distances of centrioles to nuclei
    # DistDic : dictionnary of distances with nuclei as key
    # CtrDict_B : dictionnary of nuclei-centrioles in 2048x2048
    return CtrDict, Distances, DistDic, CtrDict_B

#***************************************************************************************************
#********************************** Neural Network Functions *************************************************
#***************************************************************************************************

# creates a pretrained model
model = StarDist2D.from_pretrained('2D_versatile_fluo')

class Net(nn.Module):
    def __init__(self,hidden_layer):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(512, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, 3)
        
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm1d(hidden_layer)
        
        self.activation = nn.Softmax(1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.batch_norm1(self.conv1(x)), kernel_size=3, stride=3))
        x = F.relu(F.max_pool2d(self.batch_norm2(self.conv2(x)), kernel_size=3, stride=3))
        x = F.relu(F.max_pool2d(self.batch_norm3(self.conv3(x)), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 512)))
        x = self.batch_norm4(x)
        x = self.activation(self.fc2(x))
        
        return x
    
model1 = Net(250)
model1.load_state_dict(torch.load('modelLast'))
model1.eval()