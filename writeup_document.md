#**Traffic Sign Recognition** 

##Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_assets/traffic_signs.png "Traffic signs dataset visualization"
[image2]: ./writeup_assets/training_samples_histogram.png "Training samples histogram"
[image3]: ./writeup_assets/validation_samples_histogram.png "Validation samples histogram"
[image4]: ./writeup_assets/test_samples_histogram.png "Test samples histogram"
[image5]: ./writeup_assets/training_flip.png "Augmentation using flip "
[image6]: ./writeup_assets/flip_samples_histogram.png "Flipped samples histogram "
[image7]: ./writeup_assets/flip_augmented_histogram.png "Flipped augmented histogram "
[image8]: ./writeup_assets/rotation_samples.png "Rotation augmented samples"
[image9]: ./writeup_assets/rotation_samples_histogram.png "Rotation samples histogram"
[image10]: ./writeup_assets/augmented_dataset_histogram.png "Augmented dataset histogram"
[image11]: ./writeup_assets/Y_equalized_samples.png "Y equalized samples"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my
 [project code](https://github.com/vladpaunescu/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

Note: The cells are not numbered in incereasing order, since the process of regenerating all the data, and training the network is tedious.

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The initial traffic sign dataset is loaded in In[64] cell at Step 0 laod the data.
THe dataset size is:

* Train count 34799
* Valid count 4410
* Test count 12630


I used the numpy library to calculate summary statistics of the traffic
signs data set.

In cell 64 - Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas:

* The size of training set is 34799
* The size of validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43.

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. There are 10 samples for each class.

![alt text][image1]

The statistics of samples/category can be seen in the following histogram charts.
![alt text][image2]
![alt text][image3]
![alt text][image4]


###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

#### Augmentation using Flipping
I noticed that the dataset is unbalanced among categories.
So, one of the first ideas was to augment the data using flipping.
Unfortunately, not all traffic signs could be flipped without changing meaning.
So, I identified 4 categories of traffic signs that could be flipped:

1. some classes can be horizontally flipped.
2. others can be vertically flipped
3. others both horizontally and vertically
4, others change meaning (category) when flipped horizontally.

The code for flipping is contained in cell 22 (method definition), and cell 23 (dataset processing).

I obtained **21 569** flipped training samples using combined flipping techniques.

You can inspect some samples below. The code for sample inspection is in cell 10.
![alt text][image5]

The distribution of flipped images looks however imbalanced. The code for flip histogram is in cell 24.
![alt text][image6]

So, I concatenated the original training samples with the flipped training samples, and obtained the following distribution. From the distribution we can easuily see that there are lots of Priorit road signs. That can be explained by the fact that this traffic sign can be flipped both vertically and horizontally - it is symmetric.

The augmented dataset with flip contained **56 368 images**.


![alt text][image7]

####  Augmentation using Rotation

As a second step for augmenting the dataset, I tried to add more synthetic data to the classes with fewer elements. The maximum number of elements/class was 5670 - after flipping.
 
The idea was to slightly rotate the images for classes with fewer elements:

* if there are less than 1500 images / class: 2 rotations of +/- 5 degrees
* if there are less than 1200 images / class: 2 additional rotations of +/- 10 degrees
* if there are less than 700 images / class: 2 additional rotations of +/- 15 degrees

Using this procedure, I obtained **77 572** additional images. 

The code for rotation is located in Cell 37, and Cell 19.

You can see some of the rotations samples below: 

![alt text][image8]

Using rotation, I augmented the training set to a total of **133 940** training images.
The histogram with rotation samples is presented below:
![alt text][image9]

The final augmented dataset is presented in the histogram below:
![alt text][image10]

Now the dataset looks more balanced across classes, suitable for a deep learning approach.
Still, there are many things to improve the training accuracy, that we can do on the dataset, besides generating augmented data.

I noticed that the images are quite different in illumination. There are a lot of overexposed images, and a lot of underexposed images. So, I decided to convert them to a color space that separates the color components from the Illumination component - YUV colorspace.
Then the idea was to normalize the exposure using adaptive histogram equalization.

In **Cell 47** you can see the code for YUV conversion, and for adaptive histogram equalization.
The process is tedious so that's why the Cell names are not numbered.
In **Cell 48** a pickle the YUV data on disk to save me some time for later retrieval.

In **Cell 49** I randomly pick some samples of the Y equalized channel. You can inspect them below:
![alt text][image11]

In **Cell 61** I randomly shuffle tha assembled YUV dataset.


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in **Cell 61 ** of the IPython notebook.  


My final training set had **133940**  number of images. My validation set and test set had **4410** and **12 630** number of images.

The difference between the original data set and the augmented data set is the following.
I've flipped and rotated the training set. I've converted traingin, validation, and test set to YUV color space, and normalized the illumination Y.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the **Cell 209** of the ipython notebook. 
It is an adaptation from Pierre Sermanet and Yan LeCun - Traffic Sign Recognition with Multi-scale CNNs.

My final model consisted of the following.


| Layer         		|     Description	        					       | 
|:---------------------:|:----------------------------------------------------:| 
| Input         		| 32x32x1 Y image   							       | 
| Convolution 5x5     	| 32 maps, 1x1 stride, Valid padding, outputs 28x28x32 |
| RELU					|												       |
| Max pooling	1      	| 2x2 stride,  outputs 14x14x32				           |
| Convolution 5x5	    | 64 maps, 1x1 stride, Valid padding, outputs 10x10x64 |
| ReLU               	|            									       |
| Max pooling    		| 2x2 stride,  outputs 5x5x64					       |
| Flatten max pool 1	| Input = 14x14x32. Output = 6272.				       |
| Flatten max pool 2  	| Input = 5x5x64. Output = 1600.				       |
| Concat        		| Input: 6272 + 1600 = 7872 output				       |
| Fully connected 100 	| Input = 7872. Output = 100 					       |
| ReLU				    |           									       |
| Dropout				| 0.75 probability  							       |
| Fully connected 100	| Input = 100. Output = 100.      				       |
| ReLU   				|           									       |
| Dropout				| 0.75 probability  							       |
| Fully connected 43	| Input = 100. Output = 43.   					       |
| Softmax				| 							                           |

 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the **Cells 211, 212, 213, and 58** of the ipython notebook. 

The code is pretty standard, as the one used in the original LeNet Implementation.

To train the model, I used Adam Optimizer for adjusting the learning rate in adaptive manner.
Hyperparameters used :
* base learning rate is 0.001
* Adam Optimizer
* 30 epochs of training
* Xavier random initialization
* Input data between 0 and 1
* Dropout ratio of 0.75 (with 0.5, it won't learn)
* Batch size of 512

#### Validation Accuracy
The validation accuracy was around **0.978**.
#### Test Accuracy
The test accuracy is **0.969**.


#### Notes

I used LeNet as a baseline.
LeNet achieves ** a test accuracy **0.870** with no data augmentation.
LenNet with flip achieves **0.897.**
LeNet with rotation achievs **0.91**.



####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ** Cell 215** of the Ipython notebook.

My final model results were:
* validation set accuracy of **0.978** at epoch 30 
* test set accuracy of **0.969**

The first chosen architecture was LeNet, and it achieved only:

* 0.870 with no data augmentation.
* 0.897  with flip augmentation
* 0.91 with rotation augmentation

I've decided to use multiscale convolutional neural network that concatenates featuers from 2 pooling scale and analyze only on grayscale Y channel that was normalized. 

The dropout was used to prevent overfeating, and multiscale was chosen because the input size is small, and to capture details from 2 different scales.

The color information is not that useful, because for traffic signs the shape, and gradients are more important.

The sermanet architecture contained 108 convolutions on first layer, followed by a pooling and then 108 convs. I decided to use instead 32 convolutions on first layer and **double** the number of activations to 64 for the second convolutional layer, after pooling, because this is standard procedure. I achieved good results.

I decided to use ReLU instead of rectified tanh to prevent gradient vanishing problem.

More details about the reference paper can be found in the original [sermanet](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) .


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I tested the trained model on many traffic sign images.
Some of them are don't belong to the categories that the network was trained on.
In total there are 16 images.

Here are five German traffic signs that I found on the web

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 