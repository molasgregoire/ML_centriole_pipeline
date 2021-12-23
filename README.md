# ML_centriole_pipeline
This github contains the pipeline to detect centrioles and nuclei in fluorescence images and proceed to associate them

### Repo organization :



## Part A : Centriole detection 
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

## Part B : Nucleus detection

#### Required libraries for this part


## Part C : Centrioles-Nucleus association 

#### Required libraries for this part

