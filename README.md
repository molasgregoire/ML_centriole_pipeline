# ML_centriole_pipeline
This github contains the pipeline to detect centrioles and nuclei in fluorescence images and proceed to associate them

### Repo organization :

In the main page you will find:

## `data`:
* RPE1wt_CEP152+GTU88+PCNT_1: (goes by CEP152 in some notebook)
* RPE1wt_CEP63+CETN2+PCNT_1: (goes by CEP63 in some notebook)
* RPE1wt_CP110+GTU88+PCNT_2: (goes by CP110 in some notebook)
* DAPI: Gathered 75 DAPI images (from CEP152,CEP63,CP110)
* annotations.csv: x,y location of centroids

## `00_Explore.ipynb`: 
This notebook allows you to get familiar with the provided dataset.


## `Part A : Centriole detection`: 
This folder contains the functions, tools, and models produced for this project on detecting centriole in fluorescence images

#### Required libraries for this part
- numpy		
- os		
- pandas		
- maltplotlib	
- cv2		
- torch		
- time		
- skimage		
- scipy		

#### file description
jupyter notebooks :
- Train_a_model : this notebook aim to train a new model with a certain set of image and parameter customizable

- perform_predictions : this notebook aim to produce a csv file containing the position of the centrioles predicted on images according to a model
The resulting csv file has as columns the names of the image, the detected centrioles coordinates x and y, and the score associated to the prediction

python files :
- function : this file contains most functions that are use in other files

- cross_valisation_* : those files aim to perform a cross validation on different parameter indicated in the name of the files
(number of epoch per image, learning rate, hidden layer number, hidden layer size)

- train_model_final_specific : this file aim to train a model using the final training process developped
the loading of image is made to load the image within one folder, but it is easily editable

- threshold_figure_specific_model : this file aim to plot figures that represent the score (precision,f1,recall) for a certain model and varying the threshold to fitler prediction
this allow to observe the behavior of the results and chose the best threshold wanted

#### models
this folder contains several models that were produced, and aim to contain the new models produced by training

models description :
- CEP63_only_balanced_twice -> this model was train using the final version of the code, and with weights, on 20 images in the CEP63 images (RPE1wt_CEP63+CETN2+PCNT_1)
- CP110_only_balanced_twice -> this model was train using the final version of the code, and with weights, on 20 images in the CP110 images (RPE1wt_CP110+GTU88+PCNT_2)
- GTU88_only_balanced_twice -> this model was train using the final version of the code, and with weights, on 20 images in the GUT88 images (RPE1wt_CEP152+GTU88+PCNT_1)
- model_full -> this model is was produced with an older version of the code by training on ALL the images
- pixel14_c0 -> this model is was produced with an older version of the code by training on all the images in RPE1wt_CEP63+CETN2+PCNT_1

## `Part B : Nucleus detection`
In this folder you will find 4 notebook:
* 01_StardistExtraction.ipynb: Explain how preprocessed data of nucleus is obtained from Stardist.
* 02_StardistLabeling.ipynb: Tool to load DAPI images (which explains the `DAPI` folder in `data`), and preprocess them through Stardist. After that, the user can label them, verify the recorded labeling, and export the result as arrays. The export location is `nuc_labeling`. Is this folder you will find:
  * nucleus, labels, tags: Data extracted from Stardist without attention preprocessing.
  * nucleus1, labels1, tags1: Data extracted from Stardist with attention preprocessing.  
 
  PS: An error is displayed after cell 10, this is because we did not label all requested data again before to submit this repo. But everything works fine.
* 03_Classifier.ipynb: Further processing of extrated data: Data augmentation, standardization. Definition of the Classifier model, training, and export of it. (`modelLast`)
* 04_NucleusPrediciton.ipynb: Implementation of our model and comparison of output with "raw Stardist".

#### Required libraries for this part
* numpy
* matplotlib
* opencv
* seaborn
* skimage
* imgaug
* pytorch
* Stardist:[Installation procedure](https://github.com/stardist/stardist)


## Part C : Centrioles-Nucleus association 

#### Required libraries for this part
All previous libraries are required to run this part

Currently only work with CEP63 prediction present as a csv file in the part C directory

AssociationTask.ipynb : Jupyter file to execute task C, modify path and image to analyze accordingly
AssociationTask.py : function files that defines models & functions
modelLast : NN model file loaded using pytorch to run Nucleus detection & clssification



