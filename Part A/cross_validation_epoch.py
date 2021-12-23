###

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

import matplotlib.pyplot as plt

import scipy.ndimage.filters as filters
from skimage.feature import peak_local_max

#%%
### SEED FOR RANDOM PROCESS
np.random.seed(42)
torch.manual_seed(42)

n_cv = 5

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
        self.fc1 = torch.nn.Linear(168,n,bias=True)
        self.fc2 = torch.nn.Linear(n,n,bias=True)
        self.fc3 = torch.nn.Linear(n,n,bias=True)
        self.fc4 = torch.nn.Linear(n,n,bias=True)
        self.fc5 = torch.nn.Linear(n,n,bias=True)
        self.fc6 = torch.nn.Linear(n,n,bias=True)
        self.fc7 = torch.nn.Linear(n,n,bias=True)
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
        #x = relu(self.fc6(x))
        #x = relu(self.fc7(x))
       
        
        # out layer
        x = self.fcOut(x)
        return torch.sigmoid(x)

#create NN object and asscoiated items for training
        


#model = Pixel14().to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

criterion = nn.BCELoss()



#%%

### IMPORTATION OF THE DATA
print('importing data')
# import the file containing the positions for each centriole (csv)
data = pd.read_csv("data/annotations.csv")

# import images + the corresponding names
images,names = loadAllCtrImages(channels=[0,], format_="tif", path = path)

print( 'number of images ' + str(len(images)) )

#%%

#select n images for cross vlaidiation
indexes = [ i for i in range(len(images)) ]
np.random.shuffle(indexes)
indexes = indexes[0:n_cv]

sample_images = []
sample_names  = []

for i in indexes:
    sample_images.append( images[i] )
    sample_names.append( names[i] )
    

#%%
print('image normalization')
# normalize images
images_n = normalizeImages(sample_images)

#%%
print('ground truth generation')
# generate the ground truth -> for comparison, optionnal
images_gt = groundTruthCtr( images_n, sample_names , data , 28 ,0.85)

#%%

test_values = []
train_values = []

test_precision = []
test_f1 = []
test_recall = []

train_precision = []
train_f1 = []
train_recall = []

#%%

list_epoch = [1,2,3,4,5,6,7]

#epoch = 3
splitting_ratio = 0.8
batch_size = 1000
learning_rate = 1e-3

thr_permissivity = 0.005
pred_thr = 0.3

pre_test_values = []
pre_train_values = []

pre_test_precision = []
pre_test_f1 = []
pre_test_recall = []

pre_train_precision = []
pre_train_f1 = []
pre_train_recall = []


for epoch in list_epoch:
    # list of models
    models_optis  = []
    for i in range(n_cv):
        model = Pixel14().to(device)
        opti = torch.optim.Adam(model.parameters(), lr=learning_rate)
        models_optis.append( (model,opti) )
    for cv_id in range(n_cv):
        
        print( 'CV ' + str(cv_id) + ' starts' )
        
        #split the sets
        train_set = []
        train_names = []
        train_gt = []
        for i in range(n_cv):
            if( i != cv_id ):
                #train_set.append( (images_n[i] , images_gt[i], names[i]) )
                train_set.append( images_n[i] )
                train_names.append( names[i] )
                train_gt.append( images_gt[i] )
        
        test_set = (images_n[cv_id] , images_gt[cv_id], names[cv_id]) 
        
        
        model,optimizer = models_optis[cv_id]
        
        
        train_complete_pixel14( train_set , train_names, data ,
                  model, optimizer,
                  super_epoch = 1,epoch = epoch,gt_sensitivity = 0.85,thr_permissivity = thr_permissivity,
                  splitting_ratio = 0.8,batch_size = batch_size,learning_rate = learning_rate, use_weight = False,
                  save_name = '')
        
        #once the model is trained, we need to evaluate the loss
            
        #evaluate train set
        losses_mean_train = []
        
        p_s = []
        f_s = []
        r_s = []
        
        for idx,train_samp in enumerate(train_set):
            # get the image
            img = train_samp
            # get the ground truth of the image
            img_gt = train_gt[idx]
            
            # filter back ground for candidates
            quantile = np.quantile( img.flatten() , 1-thr_permissivity )
            _, thr = cv.threshold( img, quantile, 1., cv.THRESH_BINARY)
            
            # get all candidates coordinate in the image
            candidates = []
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if( thr[i,j] ):
                        candidates.append( (i,j) )
                        
            # form the tuples -> ( sub_image, truth ground, position )
            tuples = []
    
            for c in candidates:
                c2=(c[1],c[0])
                sub = cutAround( img, c2 , 14)
                boolean = img_gt[c]
                tuples.append( (sub,boolean,c) )
            
                
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
            
            losses = []
            #calculate the loss
            for batch_x, batch_y in dataloader_train:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    
                  # Evaluate the network (forward pass)
                prediction = model(batch_x)
                loss = criterion(prediction, batch_y)
                losses.append( loss.item() )
            
            losses_mean_train.append( np.mean(np.array(losses)) )
            
            ###
            #make the prediction for each image to test accuracy of the model
            predictions = np.zeros((2048,2048))
            for t in tuples:
                tens = torch.Tensor(np.array([ np.array( [ t[0] , sobelization( t[0] ) ])]))
                z = model(tens).item()
                predictions[t[2][0],t[2][1]] = z
                
            
                
            peaks = peak_local_max(  predictions *(predictions>pred_thr ) , min_distance=1 )
            pred_max = np.zeros((2048,2048))
            for p in peaks:
                pred_max[p[0],p[1]] = 1
            
            pred_max = ((pred_max + 1.*(img==1))>0.)*1.
            
            contours, _ = cv.findContours((pred_max>0.).astype(np.uint8),cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)
            
            #store the prediciton into dataframe format
            df_pred = pd.DataFrame( columns = [ 'image_name' , 'x','y','score'  ] )
            for i in range(0,len(contours)):
                pt = (contours[i].mean(axis=0)[0]).astype(np.int)
                #pts.append(pt)
                
                x,y,w,h = cv.boundingRect(contours[i])
                #s.append( predictions[y:y+h,x:x+w,].max()  )
                
                s_row = pd.Series([ train_samp[2] , pt[0] , pt[1] , predictions[y:y+h,x:x+w,].max() ], index=df_pred.columns)
    
                df_pred = df_pred.append(s_row,ignore_index=True) 
            
            #compute accuracy
            p_, f_, r_ = ComputePrecision(df_pred, data[ data['image_name'] == train_samp[2] ])
            
            #store values
            p_s.append(p_)
            f_s.append(f_)
            r_s.append(r_)
            
            ###
            
        pre_train_values.append(losses_mean_train)
        
        pre_train_precision.append(p_s)
        pre_train_f1.append(f_s)
        pre_train_recall.append(r_s)
            
        #evluate test set
        # get the image
        img = test_set[0]
        # get the ground truth of the image
        img_gt = test_set[1]
            
        # filter back ground for candidates
        quantile = np.quantile( img.flatten() , 0.995 )
        _, thr = cv.threshold( img, quantile, 1., cv.THRESH_BINARY)
        #gauss = cv.GaussianBlur(thr,(5,5),cv.BORDER_DEFAULT)>0
            
        # get all candidates coordinate in the image
        candidates = []
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if( thr[i,j] ):
                    candidates.append( (i,j) )
                        
        # form the tuples -> ( image, truth ground, position )
        tuples = []
    
        for c in candidates:
            c2=(c[1],c[0])
            sub = cutAround( img, c2 , 14)
            boolean = img_gt[c]
            tuples.append( (sub,boolean,c) )
            
        
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
            
        losses = []
        #calculate the loss
        for batch_x, batch_y in dataloader_train:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    
        # Evaluate the network (forward pass)
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            losses.append( loss.item() )
            
        losses_mean_test = np.mean(np.array(losses))
        pre_test_values.append( losses_mean_test )
        
        ###
        predictions = np.zeros((2048,2048))
        for t in tuples:
            tens = torch.Tensor(np.array([ np.array( [ t[0] , sobelization( t[0] ) ])]))
            z = model(tens).item()
            predictions[t[2][0],t[2][1]] = z
            
        peaks = peak_local_max(  predictions *(predictions>pred_thr ) , min_distance=1 )
        pred_max = np.zeros((2048,2048))
        for p in peaks:
            pred_max[p[0],p[1]] = 1
        
        pred_max = ((pred_max + 1.*(img==1))>0.)*1.
        
        contours, _ = cv.findContours((pred_max>0.).astype(np.uint8),cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)
        
        
        df_pred = pd.DataFrame( columns = [ 'image_name' , 'x','y','score'  ] )
        for i in range(0,len(contours)):
            pt = (contours[i].mean(axis=0)[0]).astype(np.int)
            #pts.append(pt)
            
            x,y,w,h = cv.boundingRect(contours[i])
            #s.append( predictions[y:y+h,x:x+w,].max()  )
            
            s_row = pd.Series([ train_samp[2] , pt[0] , pt[1] , predictions[y:y+h,x:x+w,].max() ], index=df_pred.columns)
            df_pred = df_pred.append(s_row,ignore_index=True) 
        
        #compute accuracy
        p_, f_, r_ = ComputePrecision(df_pred, data[ data['image_name'] == train_samp[2] ])
        
        pre_test_precision.append(p_)
        pre_test_f1.append(f_)
        pre_test_recall.append(r_)
        
        ###
        
        print( 'CV ' + str(cv_id) + ' ends' )
    
    
    test_values.append( pre_test_values )
    train_values.append( pre_train_values )
    
    test_precision.append( pre_test_precision )
    test_f1.append( pre_test_f1 )
    test_recall.append( pre_test_recall )
    
    train_precision.append( pre_train_precision )
    train_f1.append( pre_train_f1 )
    train_recall.append( pre_train_recall )

#%%

vec_x = []
vec_y_test = []
vec_y_train = []

plt.figure( figsize = (20,10))
for i in range(len(test_values)):
    vec_x.append(i)
        
    plt.plot(i,np.mean(train_values[i]),'ro',)
    vec_y_train.append(np.mean(train_values[i]))
    
    for j in train_values[i]:
        plt.plot( i , np.mean(j) , 'ro' , alpha = 0.3 )
        
    plt.plot(i+0.5,np.mean(test_values[i]),'bo')
    vec_y_test.append( np.mean(test_values[i]) )
    
    for j in test_values[i]:
        plt.plot( i+0.5 , j , 'bo' , alpha = 0.3 )

plt.plot(vec_x,vec_y_train,'r:')
plt.plot(np.array(vec_x) + 0.5 ,vec_y_test,'b:')

plt.title( 'Cross Validation Losses : Epoch'  )

plt.xlabel('Epoch')
plt.ylabel('BCE loss')

plt.plot(1,np.mean(train_values[1]),'ro',label='train')
plt.plot(1+0.5,np.mean(test_values[1]),'bo',label = 'test')
plt.legend()

plt.xticks( [ i+0.25 for i in range(len(list_epoch)) ] , list_epoch )

#%%
## F1 score

vec_x = []
vec_y_test = []
vec_y_train = []

plt.figure( figsize = (20,10))
for i in range(len(test_values)):
    vec_x.append(i)
    
    plt.plot(i,np.mean(train_f1[i]),'ro',)
    vec_y_train.append(np.mean(train_f1[i]))
    
    for j in train_f1[i]:
        plt.plot( i , np.mean(j) , 'ro' , alpha = 0.3 )
        
    plt.plot(i+0.5,np.mean(test_f1[i]),'bo')
    vec_y_test.append( np.mean(test_f1[i]) )
    
    for j in test_f1[i]:
        plt.plot( i+0.5 , j , 'bo' , alpha = 0.3 )

plt.title( 'Cross Validation F1 : Epoch'  )

plt.xlabel('Epoch')
plt.ylabel('F1')

plt.plot(1,np.mean(train_f1[1]),'ro',label='train')
plt.plot(1+0.5,np.mean(test_f1[1]),'bo',label = 'test')
plt.legend()

plt.plot(vec_x,vec_y_train,'r:')
plt.plot(np.array(vec_x) + 0.5 ,vec_y_test,'b:')

plt.xticks( [ i+0.25 for i in range(len(list_epoch)) ] , list_epoch )

#%%
## prescision score

vec_x = []
vec_y_test = []
vec_y_train = []

plt.figure( figsize = (20,10))
for i in range(len(test_values)):
    vec_x.append(i)
    
    plt.plot(i,np.mean(train_precision[i]),'ro',)
    vec_y_train.append(np.mean(train_precision[i]))
    
    for j in train_precision[i]:
        plt.plot( i , np.mean(j) , 'ro' , alpha = 0.3 )
        
    plt.plot(i+0.5,np.mean(test_precision[i]),'bo')
    vec_y_test.append( np.mean(test_precision[i]) )
    
    for j in test_precision[i]:
        plt.plot( i+0.5 , j , 'bo' , alpha = 0.3 )

plt.title( 'Cross Validation Precision : Epoch'  )

plt.xlabel('Epoch')
plt.ylabel('Precision')

plt.plot(1,np.mean(train_precision[1]),'ro',label='train')
plt.plot(1+0.5,np.mean(test_precision[1]),'bo',label = 'test')
plt.legend()

plt.plot(vec_x,vec_y_train,'r:')
plt.plot(np.array(vec_x) + 0.5 ,vec_y_test,'b:')

plt.xticks( [ i+0.25 for i in range(len(list_epoch)) ] , list_epoch )

#%%        
## recall score

vec_x = []
vec_y_test = []
vec_y_train = []

plt.figure( figsize = (20,10))
for i in range(len(test_values)):
    vec_x.append(i)
        
    plt.plot(i,np.mean(train_recall[i]),'ro',)
    vec_y_train.append(np.mean(train_recall[i]))
    
    for j in train_recall[i]:
        plt.plot( i , np.mean(j) , 'ro' , alpha = 0.3 )
        
    plt.plot(i+0.5,np.mean(test_recall[i]),'bo')
    vec_y_test.append( np.mean(test_recall[i]) )
    
    for j in test_recall[i]:
        plt.plot( i+0.5 , j , 'bo' , alpha = 0.3 )

plt.title( 'Cross Validation Recall : Epoch'  )

plt.xlabel('Epoch')
plt.ylabel('recall')

plt.plot(1,np.mean(train_recall[1]),'ro',label='train')
plt.plot(1+0.5,np.mean(test_recall[1]),'bo',label = 'test')
plt.legend()

plt.plot(vec_x,vec_y_train,'r:')
plt.plot(np.array(vec_x) + 0.5 ,vec_y_test,'b:')

plt.xticks( [ i+0.25 for i in range(len(list_epoch)) ] , list_epoch )

#%%             
param = 'epoch'
np.save( 'test_values_cv_'+param , np.array(test_values) ,  )
np.save( 'train_values_cv_'+param , np.array(train_values) ,  )

np.save( 'test_precision_cv_'+param , np.array(test_precision) ,  )
np.save( 'test_f1_cv_'+param , np.array(test_f1) ,  )
np.save( 'test_recall_cv_'+param , np.array(test_recall) ,  )

np.save( 'train_precision_cv_'+param , np.array(train_precision) ,  )
np.save( 'train_f1_cv_'+param , np.array(train_f1) ,  )
np.save( 'train_recall_cv_'+param , np.array(train_recall) ,  )



















