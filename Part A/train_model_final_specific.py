# -*- coding: utf-8 -*-
""" this file aim to apply the model training (final version) on a specific group of image """

### LIBRARY 

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

#%%
### SEED FOR RANDOM PROCESS
np.random.seed(42)
torch.manual_seed(42)

#%%
# note, the problematic images were removed
### PARAMETERS
super_epoch = 2         # number of iteration over ALL images
epoch = 5               # number of iteration for one image
gt_sensitivity = 0.85   # ground truth sensitivity (between 0 and 1, high value recommended)
thr_permissivity = 0.005# % of the brighest part of the image we keep, the lower the faster the process (1% recommended)
                        # (between 0 and 1, low value recommended) -> the lesser is the value, the more candidates
splitting_ratio = 0.9   # splitting ratio for train/test splitting
batch_size = 100        # batch size for NN training
learning_rate = 1e-3    # learning rate for NN training (1e-5 stabilize quite fast, let test with 1e-6)
channels = [0,]
                        #+rotation
save_name = 'CP110_only_balanced_twice'# the name for the file saved at the end

#%%

### IMPORTATION OF THE DATA
print('importing data')
# import the file containing the positions for each centriole (csv)
data = pd.read_csv("../data/annotations.csv")

# import images + the corresponding names
#images,names = loadAllCtrImages(channels=channels, format_="tif", path = path)

###
## we load the 25 but only train on 20 to test on the 5 remaining
# Load all CEP63 (or other folder) images
#path = '../data/RPE1wt_CEP63+CETN2+PCNT_1/CEP63/tif'
# CP110
path = '../data/RPE1wt_CP110+GTU88+PCNT_2/CP110/tif'

images = []
names = [f for f in listdir(path) if isfile(join(path, f))]

for n in names:
    images.append( cv.imread( path + '/' + n , cv.IMREAD_UNCHANGED) )

names = [n[:-4] for n in names]

print( 'number of images ' + str(len(images)) )

#%%
spl_nb = 20

indexes_spl = [ i for i in range(len(images)) ]
np.random.shuffle( indexes_spl )
#%%
spl_images = []
spl_names = []

for i in indexes_spl[:spl_nb]:
    spl_images.append( images[i] )
    spl_names.append( names[i] )

#%%
print('image normalization')
# normalize images
images_n = []
for img in spl_images:
    img = img.astype(np.float)
    img -= img.min()
    img /= img.max()
    images_n.append(img)

#%%
print('ground truth generation')
# generate the ground truth 
images_gt = groundTruthCtr( images_n, spl_names , data , 28 ,gt_sensitivity)

#%%

### OUR NEURAL NETWORK

# device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class
class Pixel14(torch.nn.Module):
    def __init__(self):
        """From: LeCun et al., 1998. Gradient-Based Learning Applied to Document Recognition"""
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 21, kernel_size=2)
        self.conv2 = torch.nn.Conv2d(21, 42, kernel_size=2)
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
        # test avec 5layers
        
        # out layer
        x = self.fcOut(x)
        return torch.sigmoid(x)

#create NN object and asscoiated items for training
model = Pixel14().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()
#BECloss redefined within the loops as we want to integrate weights depending of the images

#%%

### RUN EVERYTHING

# create a list of index for images in order to randomize their order
indexes = [ i for i in range(len(images_n)) ]

# start time counter
start = time.time()
print( 'Iterations start here' )

#iteration over ALL images
for sup_e in range(super_epoch):
    #shuffle images order
    np.random.shuffle( indexes )
    count = 0
    #itrations over images
    for idx in indexes:
        # get the image
        img = images_n[idx]
        
        # filter back ground for candidates
        quantile = np.quantile( img.flatten() , 1-thr_permissivity )
        _, thr = cv.threshold( img, quantile, 1., cv.THRESH_BINARY)
        
        
        # get all candidates coordinate in the image
        candidates = []
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if( thr[i,j] ):
                    candidates.append( (i,j) )
                    
        # get the ground truth of the image
        img_gt = images_gt[idx]
        
        ###
        #BALANCING
        count_gt = img_gt.sum()
        count_all = thr.sum()
        propotion_gt = max(1,count_gt)/count_all  
        
        weight = torch.tensor([  1/propotion_gt ])
        criterion = nn.BCELoss( weight = weight )
        ###
        
        # form the tuples -> ( sub_image, truth ground, position )
        tuples = []

        for c in candidates:
            c2=(c[1],c[0])
            sub = cutAround( img, c2 , 14)
            boolean = img_gt[c]
            tuples.append( (sub,boolean,c) )
        
        ###
        #duplicate ALL tuples with rotations of images
        tmp_tuples = []
        for rota in range(1,4):
            for t in tuples:
                tmp = np.rot90(t[0],k=rota)
                tmp_tuples.append( (tmp,t[1],t[2]) )
            
        tuples = tuples + tmp_tuples
        ###
            
        #shuffling
        np.random.shuffle( tuples )
        
        #split tuples into train / test sets
        middle = int( len(tuples) * splitting_ratio )
        tuples_train = tuples[:middle]
        tuples_test = tuples[middle:]
            
        # build tensors and then dataloaders from the tuples
        ##TRAIN
        tensor_x = torch.Tensor(np.array([np.array( [ i[0] , sobelization(i[0]) ]) for i in tuples_train])) # transform to torch tensor
        tensor_y = torch.Tensor(np.array([np.array([i[1]]) for i in tuples_train]))
        
        #then create the Dataloader
        my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
        dataloader_train = DataLoader(my_dataset,batch_size=batch_size,shuffle=True) # create your dataloader
        
        ##TEST
        tensor_x = torch.Tensor([np.array( [i[0] , sobelization(i[0]) ]) for i in tuples_test]) # transform to torch tensor
        tensor_y = torch.Tensor([np.array([i[1]]) for i in tuples_test])
        
        #then create the Dataloader
        my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
        dataloader_test = DataLoader(my_dataset,batch_size=batch_size,shuffle=False) # create your dataloader

        # train the NN model
        train(model, criterion, dataloader_train, dataloader_test, optimizer, epoch)

        count += 1
        
        print( 'super epoch #' + str(sup_e) + ' image #' + str(count) + 
              ' in ' + str((time.time() - start)//60) + ' minutes ' + str((time.time() - start)%60) + ' seconds ' )
    print( 'super epoch #' + str(sup_e) + str((time.time() - start)//60) + ' minutes ' + ' in ' + str((time.time() - start)%60) + 'seconds' )

#%%

### SAVE THE MODEL

torch.save(model, 'models/' + save_name)




























