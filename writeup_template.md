# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./Other/visualization1.png "Visualization1"
[image2]: ./Other/visualization2.png "Visualization2"
[image3]: ./Other/visualization3.png "Visualization3"
[image4]: ./TestImage/001.jpg "Traffic Sign 1"
[image5]: ./TestImage/002.jpg "Traffic Sign 2"
[image6]: ./TestImage/003.jpg "Traffic Sign 3"
[image7]: ./TestImage/004.jpg "Traffic Sign 4"
[image8]: ./TestImage/005.jpg "Traffic Sign 5"
[image9]: ./Other/before.png "Before"
[image10]: ./Other/after.png "After"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! And my project code is included in the same zip file.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 * 32 * 3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. Below are bar charts showing the distribution of training set, validation set and testing set.

![Visualization of training set][image1]
![Visualization of validation set][image2]
![Visualization of test set][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

How I preprocessed the image data was to normalize the images and convert them to grayscale. Normalization can make the contrast of images better. Because I wanted the model to focus on the shape instead of the color of the image so I also converted them to grayscale. Below are images before and after preprocessing.

##### Before
![Before][image9]
##### After
![After][image10]  

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6	 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16	 				|
| Flatten		      	| 5x5x16 = 400					 				|
| Fully connected		| 400 to 120        							|
| Fully connected		| 120 to 84        								|
| Fully connected		| 84 to 43     									|
| Softmax				| 43 possible final outputs						|
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer, set batch size to 128 and number of epochs to 20 , and let learning rate equal to 0.001. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 93.9% 
* test set accuracy of 91.5%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

I chose a well known architecture called Lenet-5.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The fourth and fifth image might be difficult to classify because they are either not or not.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        				|     Prediction	        					| 
|:-------------------------------------:|:---------------------------------------------:| 
| Yield		      						| Yield		  									| 
| Stop     								| Children crossing 							|
| Right-of-way at the next intersection	| Right-of-way at the next intersection			|
| Dangerous curve to the right	      	| No entry					 					|
| Pedestrians							| General caution      							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 8th cell of the Ipython notebook.

For all the image, the model is sure that all the predictions it makes are sure that this is a yield sign (probability of 1.0), and the image does contain a yield sign. The top five soft max probabilities were

| Probability         	|     Prediction	        				| 
|:---------------------:|:-----------------------------------------:| 
| 1.00         			| Yield   									| 
| 0.00     				| Speed limit (20km/h) 						|
| 0.00					| Speed limit (30km/h)						|
| 0.00	      			| Speed limit (50km/h)						|
| 0.00				    | Speed limit (60km/h)      				|


For the second image ... 

| Probability         	|     Prediction	        				| 
|:---------------------:|:-----------------------------------------:| 
| 1.00         			| Children crossing   						| 
| 0.00     				| Speed limit (20km/h) 						|
| 0.00					| Speed limit (30km/h)						|
| 0.00	      			| Speed limit (50km/h)						|
| 0.00				    | Speed limit (60km/h)      				|

For the third image ... 

| Probability         	|     Prediction	        				| 
|:---------------------:|:-----------------------------------------:| 
| 0.943        			| Right-of-way at the next intersection   	| 
| 0.026     			| General caution	 						|
| 0.022					| Pedestrians								|
| 0.004	      			| Beware of ice/snow						|
| 0.001				    | Double curve			      				|

For the fourth image ... 

| Probability         	|     Prediction	        				| 
|:---------------------:|:-----------------------------------------:| 
| 1.00         			| No entry   								| 
| 0.00     				| Speed limit (20km/h) 						|
| 0.00					| Speed limit (30km/h)						|
| 0.00	      			| Speed limit (50km/h)						|
| 0.00				    | Speed limit (60km/h)      				|

For the fifth image ... 

| Probability         	|     Prediction	        				| 
|:---------------------:|:-----------------------------------------:| 
| 1.00         			| General caution   						| 
| 0.00     				| Speed limit (20km/h) 						|
| 0.00					| Speed limit (30km/h)						|
| 0.00	      			| Speed limit (50km/h)						|
| 0.00				    | Speed limit (60km/h)      				|





### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?