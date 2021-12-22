# -*- coding: utf-8 -*-
""" this file contains most functions useful for this project """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage import measure, color
from os import listdir 
from os.path import isfile, join

#pytorch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

#counting time
import time

import scipy.ndimage.filters as filters
from skimage.feature import peak_local_max



#hello world -> use to check if this file is correctly imported
def helloWorld():
    print('hello world !')

# Import images (here only those with centrioles <- not DAPI)
#set path to data folder
path = 'data/'
#give teh name for the different sub folders
options = [["RPE1wt_CEP152+GTU88+PCNT_1","DAPI","CEP152","GTU88","PCNT"], #Channel 0 : C0,C1,C2,C3
           ["RPE1wt_CEP63+CETN2+PCNT_1","DAPI","CEP63","CETN2","PCNT"], #Channel 1 : C0,C1,C2,C3
           ["RPE1wt_CP110+GTU88+PCNT_2","DAPI","CP110","GTU88","PCNT"]] #Channel 2 : C0,C1,C2,C3

#***************************************************************************************************
#********************************** ACCESS PICTURE *************************************************
#***************************************************************************************************

""" Get paths for ALL centriol folder in each channels """
def generatePathsCtr(channels=[], format_="tif", path = path):
    paths=[]
    # generate path strings
    for i in channels:
        paths += [(path + 
            options[i][0] + "/" +
            options[i][C] + "/" +
            format_ )
            for C in range(2,5)]
    return paths

""" return ALL images and their name in each channel  """
def loadAllCtrImages(channels=[], format_="tif", path = path):
    #get path to folders (channels)
    paths = generatePathsCtr(channels, format_, path)
    names = []
    images = []
    
    #iterate on path and get each image in the folder (+their names)
    for p in paths:
        tmp = [f for f in listdir(p) if isfile(join(p, f))]
        
        for n in tmp:
            images.append( cv.imread( p + '/' + n , cv.IMREAD_UNCHANGED) )
        #remove '.format' at the end of file name
        tmp = [ i[:-4] for i in tmp]
        names+=tmp
        
    return images, names

""" return ALL images and their name in the folder indicated by the path (tif format) """
def loadAllImagesInPath( path ):
    images = []
    #get the files name in the folder
    names = [f for f in listdir(path) if isfile(join(path, f))]
    #get each image
    for n in names:
        images.append( cv.imread( path + '/' + n , cv.IMREAD_UNCHANGED) )
    # remove the suffix (.tif) of the names
    names = [n[:-4] for n in names]
    
    return images, names

""" extract the sub images of known centrioles in images """
# side -> size of the side of the square surrounding the centriol position
# shift -> if True, it will randomly shift the image (the centriole position will always stay within the square)
# return a list of images (np.array), and a list containing the local positions of centriols within those images
def extractCtr( images, names , df , side , shift = False ):
    centrioles = []
    local_coors = []
    halfSide = int(side/2)
    
    #iteratre on images
    for i in range(len(images)):
        #get only the information for the image of interest
        sub = df[df['image_name']==names[i]]
        
        #iterate on each known coordinate
        for _,c in sub.iterrows():
            x=c.x
            y=c.y
            shift_x=0
            shift_y=0
            #shifting
            if( shift ):
                shift_x = np.random.randint(0,halfSide)*(np.random.randint(0,2)*2 -1)
                shift_y = np.random.randint(0,halfSide)*(np.random.randint(0,2)*2 -1)
                x += shift_x
                y += shift_y
            
            #get the position of the sides of the cut square
            xMinus = x-halfSide
            xPlus  = x+halfSide
            yMinus = y-halfSide
            yPlus  = y+halfSide
                        
            ###
            #check and incorporate other centriole in the patch
            dfTmp = sub[sub['x']>xMinus]
            dfTmp = dfTmp[dfTmp['x']<xPlus]
            dfTmp = dfTmp[dfTmp['y']>yMinus]
            dfTmp = dfTmp[dfTmp['y']<yPlus]          
            ###
            lc = []
            for _,c in dfTmp.iterrows():
                lc.append( [c.x-x+halfSide , c.y-y+halfSide] )
            local_coors.append(lc)
            
            #check if their is no value out of range
            if( (xMinus >=0) & (yMinus >=0) 
             & (xPlus <= images[i].shape[1] )
             & (yPlus <= images[i].shape[0] )):
                centrioles.append( images[i][yMinus:yPlus,xMinus:xPlus,] )
                
            else: #management of border cases with zero padding
                proxy=np.zeros((side,side))
                for p1 in range(side):
                    for p2 in range(side):
                        if( (xMinus+p1 >0) & (yMinus+p2 >0) 
                         & (xMinus+p1 < images[i].shape[1] )
                         & (yMinus+p2 < images[i].shape[0] )):
                            proxy[p2,p1]=images[i][yMinus+p2,xMinus+p1,]
                centrioles.append(proxy)
    return centrioles, local_coors

""" train the model automatically acoordingly to the parameter given """
# model -> pytorch model to be trained
# criterion -> criterion function (BCEloss in this case)
# dataset_train -> pytorch dataset used to train the model
# dataset_test  -> pytorch dataset used to check the result of the train at each step
# otpimizer -> pytorch optimzer
# num_epoch -> the number of epoch = the number of time the model is trained on the data
def train(model, criterion, dataset_train, dataset_test, optimizer, num_epochs):
    
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Starting training")
    for epoch in range(num_epochs):
        # Train an epoch
        model.train()
        for batch_x, batch_y in dataset_train:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

      # Evaluate the network (forward pass)
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
      
      # Compute the gradient
            optimizer.zero_grad()
            loss.backward()

      # Update the parameters of the model with a gradient step
            optimizer.step()

    # Test the quality on the test set
        model.eval()
        accuracies_test = []
        for batch_x, batch_y in dataset_test:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

      # Evaluate the network (forward pass)
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            accuracies_test.append( loss )

        print("Epoch {} | mean criterion: {:.5f}".format(epoch, sum(accuracies_test).item()/len(accuracies_test)))


""" return the 2d (on x and y) sobel transformation of an image """
def sobelization(img, step=3):
    sobel_x = np.absolute(cv.Sobel(img,cv.CV_64F,1,0,step))
    sobel_y = np.absolute(cv.Sobel(img,cv.CV_64F,0,1,step))
    sobel_xy= (sobel_x+sobel_y)/2
    sobel_xy-=sobel_xy.min()
    if( sobel_xy.max() > 0 ):
        sobel_xy/=sobel_xy.max()
    return sobel_xy

""" given a set of images and teh information associated, generate a list of ground truth images where 0 is background and 1 is a centriole """
# t -> threshold ~ how permissive we are on 'what a centriole is' (ground truth sensitivity)
# side -> the side of the cutting around a centriole to normalize (28 is good)
# names -> the names of the images, needed to retrieve the information accordingly in the dataframe
# df -> the dataframe containing the centrioles annotation for each images
def groundTruthCtr( images, names , df , side ,t):
    images_gt = []
    halfSide = int(side/2)
    
    #ieterate on each image
    for i in range(len(images)):
        #get the inforamtion only for the image of interest
        sub = df[df['image_name']==names[i]]
        
        #create an empty map
        gt = np.zeros( images[i].shape )
        
        #iterate on each know position
        for _,c in sub.iterrows():
            tmp_centriole = 0
            
            #position of the centriole
            x=c.x
            y=c.y
            #position of the sides of the square cut around the centriole
            xMinus = x-halfSide
            xPlus  = x+halfSide
            yMinus = y-halfSide
            yPlus  = y+halfSide
                        
            ###
            #check and incorporate other centriole in the patch
            dfTmp = sub[sub['x']>xMinus]
            dfTmp = dfTmp[dfTmp['x']<xPlus]
            dfTmp = dfTmp[dfTmp['y']>yMinus]
            dfTmp = dfTmp[dfTmp['y']<yPlus]          
            ###
            lc = []
            for _,c in dfTmp.iterrows():
                lc.append( [c.x-x+halfSide , c.y-y+halfSide] )
            
            #check if their is no value out of range, if no directly do the cut
            if( (xMinus >=0) & (yMinus >=0) 
             & (xPlus <= images[i].shape[1] )
             & (yPlus <= images[i].shape[0] )):
                 tmp_centriole = images[i][yMinus:yPlus,xMinus:xPlus,] 
                
            else: #management of border case with zero padding
                proxy=np.zeros((side,side))
                for p1 in range(side):
                    for p2 in range(side):
                        if( (xMinus+p1 >0) & (yMinus+p2 >0) 
                         & (xMinus+p1 < images[i].shape[1] )
                         & (yMinus+p2 < images[i].shape[0] )):
                            proxy[p2,p1]=images[i][yMinus+p2,xMinus+p1,]
                tmp_centriole = proxy
                
            #apply local normalization + threshold filter -> rhis allow to have more than one pixel per centriole
            tmp = tmp_centriole
            tmp = np.array(tmp).astype(np.float)
            tmp -= np.min(tmp)
            tmp /= np.max(tmp)
            _, threshold = cv.threshold(tmp, t, 1., cv.THRESH_BINARY)
            #add of the default coordinate, so annotated centriole at least have one pixel
            for j in lc:
                threshold[j[1],j[0]]=1
            
            #then we repatch the small patch within the whole map        
            for p1 in range(threshold.shape[0]):
                for p2 in range(threshold.shape[1]):
                    if( (xMinus+p1 >0) & (yMinus+p2 >0) 
                         & (xMinus+p1 < images[i].shape[1] )
                         & (yMinus+p2 < images[i].shape[0] )):
                        gt[ p2+yMinus,p1+xMinus ] += threshold[p2,p1]
        
        gt = 1*(gt>0)
        images_gt.append(gt)
                
    return images_gt
    
""" cut image around a certain position (+0 padding on border) """
# img   -> the image
# pos   -> postion of the center aorund which we will cut
# side  -> side size of the patch (square)
def cutAround( img, pos , side ):
    halfSide = int(side/2)
    
    x = pos[0]
    y = pos[1]
    #position of the sides of the square cut around the position
    xMinus = x-halfSide
    xPlus  = x+halfSide
            
    yMinus = y-halfSide
    yPlus  = y+halfSide
    
    #check if their is no value out of range, if no directly do the cut
    if( (xMinus >=0) & (yMinus >=0) 
             & (xPlus <= img.shape[1] )
             & (yPlus <= img.shape[0] )):
        return img[yMinus:yPlus,xMinus:xPlus,]
    else: #management of border case with zero padding
        proxy=np.zeros((side,side))
        for p1 in range(side):
            for p2 in range(side):
                if( (xMinus+p1 >0) & (yMinus+p2 >0) 
                         & (xMinus+p1 < img.shape[1] )
                         & (yMinus+p2 < img.shape[0] )):
                    proxy[p2,p1]=img[yMinus+p2,xMinus+p1,]
        return proxy

""" check for EACH pixel in an image if it is the maxium compare to its 9 neighbors """    
def filterLocalMax( arr ):
    copie = arr.copy()
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val_local = arr[i,j]
            boolean = False
            #Ouest
            if( (i-1 >= 0) and ( arr[i-1,j] > val_local ) ): boolean =True
            #Est
            if( (i+1 < arr.shape[0]) and ( arr[i+1,j] > val_local ) ): boolean =True
            #nord
            if( (j-1 >= 0) and ( arr[i,j-1] > val_local ) ): boolean =True
            #sud
            if( (j+1 < arr.shape[1]) and ( arr[i,j+1] > val_local ) ): boolean =True
            ## coins 
            #nord ouest
            if( (i-1 >= 0) and (j-1 >= 0) and ( arr[i-1,j-1] > val_local )  ): boolean =True
            #nord est
            if( (i+1 < arr.shape[0]) and (j-1 >= 0) and ( arr[i+1,j-1] > val_local )  ): boolean =True
            #sud ouest
            if( (i-1 >= 0) and (j+1 < arr.shape[1]) and ( arr[i-1,j+1] > val_local )  ): boolean =True
            #sud est
            if( (i+1 < arr.shape[0]) and (j+1 < arr.shape[1]) and ( arr[i+1,j+1] > val_local )  ): boolean =True
            
            
            if(boolean):copie[i,j]=0
    
    return copie
    
""" this function take a list of images and the information associated, and the pytorch model to be trained """
# images            -> a list of images (np.array format)
# names             -> the names associated with the images
# df                -> the data frame containing the information on the images (the centrioles positions)
# model             -> pytorch model
# optimizer         -> pytorch optimizer
# criterion         -> pytorch criterion (BCE loss recommended in this case) -> defined by default now
# super_epoch       -> number of time the model will be trained over ALL images
# epoch             -> number of time the model will be trained when passing on a single images
# gt_snesitivity    -> see groundTruthCtr
# thr_permissivity  -> % of the most intense pixel that will be kept (aka background suppression)
# splitting_ratio   -> splitting ration of images data in train/test set
# batch_size        -> batch_size for the NN training (the higher the faster)
# learning_rate     -> learning rate for the NN training
# use_weight        -> if true, the model will take in account the imbalance between true/false values within an image
# save_name         -> the name (or rather path) to save the model at the end of the training, if no name is given the model is not saved
def train_complete_pixel14( images , names, df ,
                  model, optimizer,#criterion,
                  super_epoch = 1,epoch = 3,gt_sensitivity = 0.85,thr_permissivity = 0.01,
                  splitting_ratio = 0.8,batch_size = 100,learning_rate = 1e-3, use_weight = True,
                  save_name = ''):
    print('image normalization')
    # normalize images
    images_n = normalizeImages( images )
    
    print('ground truth generation')
    # generate the ground truth 
    images_gt = groundTruthCtr( images_n, names , df , 28 ,gt_sensitivity)    
    
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
            if(use_weight):
                count_gt = img_gt.sum()
                count_all = thr.sum()
                propotion_gt = max(1,count_gt)/count_all  
                
                weight = torch.tensor([  1/propotion_gt ])
                criterion = nn.BCELoss( weight = weight )
            else: criterion = nn.BCELoss()
            ###
            
            # form the tuples -> ( sub_image, truth ground, position )
            tuples = []
    
            for c in candidates:
                c2=(c[1],c[0])
                sub = cutAround( img, c2 , 14)
                boolean = img_gt[c]
                tuples.append( (sub,boolean,c) )
            
            ###
            #duplicate ALL tuples with rotations of images (90°,180°,270°)
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
                  ' in ' + str((time.time() - start)//60) + ' minutes ' + str(round((time.time() - start)%60,0)) + ' seconds ' )
        print( 'super epoch #' + str(sup_e) + str((time.time() - start)//60) + ' minutes ' + ' in ' + str(round((time.time() - start)%60,0)) + 'seconds' )
    
    ### SAVE THE MODEL
    if( len(save_name) > 0 ):
        torch.save(model,  save_name)    
    
    
""" this function perform the predictions on a list of images using a pytorch model, and produce the csv containing the predicted positions of centrioles """
# images            -> a list of images (np.array format)
# names             -> the names associated with the images
# model             -> pytorch model
# thr_permissivity  -> % of the most intense pixel that will be kept (aka background suppression)
# pred_thr          -> threshold for prediction (remove low value and keep above values for contours)
# save_name         -> the name (or rather path) to save the model at the end of the training, if no name is given the model is not saved
def generate_csv_pixel14( images, names, model,
                         thr_permissivity = 0.01, pred_thr = 0.5,
                         save_name = '' ):
    

    # normalize images
    images_n = normalizeImages( images )

    df = pd.DataFrame( columns = [ 'image_name' , 'x','y','score'  ] )
    print('iterations start here')
    for idx, img in enumerate(images_n):
        #reove background
        quantile = np.quantile( img.flatten() , 1-thr_permissivity )
        _, threshold = cv.threshold(img, quantile, 1., cv.THRESH_BINARY)
        gauss = cv.GaussianBlur(threshold,(5,5),cv.BORDER_DEFAULT) > 0
        
        #iterate on interesting positions -> generate predictions and store them
        predictions = np.zeros((2048,2048))
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if( gauss[i,j] ):
                    sub = cutAround( img, (j,i) , 14)
                    tens = torch.Tensor(np.array([ np.array( [ sub , sobelization( sub ) ])]))
                    z = model(tens).item()
                    predictions[i,j] = z
                    
        #get the punctual maxima of predictions
        #-> produce a 0/1 map that will be read by cv.findContours
        peaks = peak_local_max(  predictions *(predictions>pred_thr ) , min_distance=1 )
        pred_max = np.zeros((2048,2048))
        for p in peaks:
            pred_max[p[0],p[1]] = 1
        pred_max = ((pred_max + 1.*(img==1))>0.)*1.
        
        #get each contours detectect as centrioles
        contours, _ = cv.findContours((pred_max>0.).astype(np.uint8),cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)
    
        #for each contours detected, store the center position and the score associated
        for i in range(0,len(contours)):
            pt = (contours[i].mean(axis=0)[0]).astype(np.int)            
            x,y,w,h = cv.boundingRect(contours[i])
            s_row = pd.Series([ names[idx] , pt[0] , pt[1] , predictions[y:y+h,x:x+w,].max() ], index=df.columns)
    
            df = df.append(s_row,ignore_index=True)
        
        print( 'image #'+ str(idx) + ' done')    
    
    if( len(save_name) > 0 ):
         df.to_csv( save_name )
    
    return df
    
    
'''Linear Assignment method to compare predictions to ground truth'''
def LinearAssignment(truth_df, pred_df):
    #truth_df : pandas dataframe of gt
    #pred_df : pandas df of predictions
    truth_x = truth_df['x'].tolist()
    truth_y = truth_df['y'].tolist()

    pred_x = pred_df['x'].tolist()
    pred_y = pred_df['y'].tolist()

    #association dict
    #-1 values correspond to unassigned truth
    associations = {i : -1 for i in range(len(truth_x))}
    unassignedCtr = []
    unassignedPred = [j for j in range(0,len(pred_x))]

    #Matrix of distances betweens each gt-prediction
    matrix = np.zeros([len(truth_df.x), len(pred_df.x)])
    for i in range(0,len(matrix[:,0])):
        for j in range(0,len(matrix[0,:])):
            dist = np.linalg.norm([truth_x[i] - pred_x[j],truth_y[i] - pred_y[j]])
            matrix[i,j] = dist
    

    #Application of hungarian algorithm
    '''Issue of not detecting lowest distance at the first time'''
    #iterate through rows to reduce the matrix
    matrix_ = matrix.copy()
    for i in range(0, len(matrix_[:,0])):
        matrix_[i,:] -= np.min(matrix_[i,:])

    #associate truth point to proximal predictions
    gt, pred = np.where((matrix_ == 0) == 1)
    for t, p in zip(gt, pred):
        if associations[t] == -1:
            if p not in associations.values():
                associations[t] = p
            if p in associations.values():
                comp = [list(associations.keys())[list(associations.values()).index(p)]]
                for c in comp:
                    if matrix[t,p] < matrix[c, p]:
                        associations[c] = -1
                        associations[t] = p
    
    #Fills unassigned points lists
    for k, v in associations.items():
        if v == -1:
            unassignedCtr.append(k)
        if v != -1:
            unassignedPred.remove(v)
    return associations, unassignedCtr, unassignedPred,  matrix


'''Compute precision, f1 score and recall for 1 image given dataframes'''
def ComputePrecision(truth_df, pred_df):
    Ass, uT, uP, Amat = LinearAssignment(truth_df, pred_df)
    TP = 0
    for k, v in Ass.items():
        if v != -1:
            TP+=1
    FP = len(uP)
    FN = len(uT)
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    f1 = TP / (TP + 0.5 * (FP + FN))
    return precision, f1, recall

    
""" take a list of images (array) and normalize them between 0 and 1 using min max method """
def normalizeImages( images ):
    images_n = []
    for img in images:
        img = img.astype(np.float)
        img -= img.min()
        img /= img.max()
        images_n.append(img)  
    return images_n

""" generate a list of prediction map (list of array) given images to analyze and a pixel14 model """
def generatePredictions( images_n, model, thr_permissivity ):
    
    list_predictions = []
    for idx, img in enumerate(images_n):
        #remove background
        quantile = np.quantile( img.flatten() , 1-thr_permissivity )
        _, threshold = cv.threshold(img, quantile, 1., cv.THRESH_BINARY)
        gauss = cv.GaussianBlur(threshold,(5,5),cv.BORDER_DEFAULT) > 0
        
        #iterate on interesting positions -> generate predictions
        predictions = np.zeros((2048,2048))
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if( gauss[i,j] ):
                    sub = cutAround( img, (j,i) , 14)
                    tens = torch.Tensor(np.array([ np.array( [ sub , sobelization( sub ) ])]))
                    z = model(tens).item()
                    predictions[i,j] = z
        list_predictions.append(predictions)
        
    return list_predictions
    
""" generate a Dataframe containing the centrioles position predicted for each prediction provided """
# list_predictions -> list of prediction map (to generate prediction map using images and models, use generatePredictions)
# names            -> names of the images, must be the same length as list prediction
# thr              -> the threshold to select prediction in the map
def generateDataFrame( list_predictions, names, thr ):
    
    df = pd.DataFrame( columns = [ 'image_name' , 'x','y','score'  ] )
    
    for i,p in enumerate(list_predictions):
        
        #get the punctual maxima of predictions
        peaks = peak_local_max(  p *(p>thr ) , min_distance=1 )
        pred_max = np.zeros((2048,2048))
        for peak in peaks:
            pred_max[peak[0],peak[1]] = 1
                
            pred_max = ( pred_max >0.)*1.
            
        #get each contours detectect as centrioles
        contours, _ = cv.findContours((pred_max>0.).astype(np.uint8),cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)
        
        
        for j in range(0,len(contours)):
            pt = (contours[j].mean(axis=0)[0]).astype(np.int)
                
            x,y,w,h = cv.boundingRect(contours[j])
            
            s_row = pd.Series([ names[i] , pt[0] , pt[1] , p[y:y+h,x:x+w].max() ], index=df.columns)
        
        df = df.append(s_row,ignore_index=True)
    
    return df

""" this function aim to determine the 'best' threshold to analyze images prediction (the threshold with the highest score = prescision+f1+recal) """
# images            -> list of images
# names             -> list of names associated with the images
# data              -> dataframe containing the true centrioles position for each images
# thr_permissivity  -> quanitle selection of the foreground
# step              -> interval between final threshold values to be tested
# max/min           -> maximum and minimum values of the space analyzed
def findBestThr( images, names, data, model, thr_permissivity , step,  max_ = 1, min_ =0 ):
    #nromalization
    images_n = normalizeImages( images )
    
    #iterate on images, generate predictions, and store them for applying threshold after    
    list_predictions = generatePredictions( images_n, model, thr_permissivity )
    
    #make a list of threshold according to step
    vec_threshold = np.arange( min_+step,max_-step,step )
    
    #lists to store values
    mean_p = []
    mean_f = []
    mean_r = []
    
    #itereate for each threshold
    for t in vec_threshold:
        #lists to store values
        p_s=[]
        f_s=[]
        r_s=[]
        #iterate on predictions, filter them using the threshold t, and compute the score by comparison to true values
        for i,p in enumerate(list_predictions):
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
                
                s_row = pd.Series([ names[i] , pt[0] , pt[1] , p[y:y+h,x:x+w,].max() ], index=df.columns)
        
                df = df.append(s_row,ignore_index=True)
            
            p_, f_, r_ = 0,0,0
            if( len(df)==0 ):
                if( len( data[ data['image_name']==names[i] ] ) == 0 ):
                    p_, f_, r_ = 1,1,1 #empty image without prediction -> 'prefect case'
                else:p_, f_, r_ = 0,0,1 #no prediction with non empty image -> 'worst case'
                
            else:
                p_, f_, r_ = ComputePrecision( df, data[ data['image_name'] == names[i] ])

            p_s.append(p_)
            f_s.append(f_)
            r_s.append(r_)
         #compute the mean for each score, over all images  
        mean_p.append(np.mean(p_s))
        mean_f.append(np.mean(f_s))
        mean_r.append(np.mean(r_s))
    #determine the 'best' score and the thresholdassociated
    max_score = 0
    max_idx = 0
    for i in range(len(vec_threshold)):
        score = mean_p[i]+mean_f[i]+mean_r[i]
        if( score > max_score ):
            max_score = score
            max_idx = i
    #return the 'best' threshold and the scores associated
    return vec_threshold[max_idx], mean_p[max_idx], mean_f[max_idx], mean_r[max_idx]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    