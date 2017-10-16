#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_imgs/distribution.png "Visualization"
[image2]: ./writeup_imgs/grayscale.png "Grayscaling"
[image3]: ./writeup_imgs/augmented-imgs.png "Random Noise"
[image4]: ./writeup_imgs/s01.jpg "No Passing"
[image5]: ./writeup_imgs/s02.jpg "No Entry"
[image6]: ./writeup_imgs/s03.jpg "Stop"
[image7]: ./writeup_imgs/s04.jpg "60kph"
[image8]: ./writeup_imgs/s05.jpg "Roundabout"

You're reading it! and here is a link to my [project code](https://github.com/desi-joe/CarND-Traffic-Sign-Classifier-Project)

###Data Set Summary & Exploration

####1. Basic summary of the data set

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed based on classifications. In this data set we have 43 different signs.  

![alt text][image1]

###Design and Test a Model Architecture

####1. Data Pre-process
As a first step, I decided to convert the images to grayscale because it's easier to work with gray scale images. 

I have use OpenCv library to convert color (RGB) images to gray scale images. 

######cv2.cvtColor(imgs[i], cv2.COLOR_RGB3GRAY) 


![alt text][image2]

As a last step, I normalized the image data because with data in the same scale our network trains better.  

I used the following code to normalize pixel values between -1 and 1. 

######imgs[i] / 127.5 - 1

If the pixel value is 0 then, 0 / 127.5 - 1 => -1
and if the pixel value is 255 then, 255 / 127.5 - 1 => 1


I decided to generate additional data because with a large dataset I can train the network better and get better accuracy. Also, it represents real life scenarios where a sign can be at an angle or crooked. 
For each image I added the following additional images


1. Image from left prospective
2. Image from right prospective
3. Image from up prospective
4. Image from down prospective
 

Here is an example of an original image and an augmented image:

![alt text][image3]
 


####2. Architecture

I used the well known LeNet model without any significat modifications. The LeNet architecture was first introduced by LeCun et al. in 1998. It's a very good architecture for recognizing hand writter numbers and it also works well for recognizing images. 

Here is the LeNet architecture with dropouts added. 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray Scale image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	 2x2     	| 28x28x6 in, 14x14x6 out 				|
| Convolution 5x5	    | 14x14x6 in, 10x10x16 out      									|
| RELU		|        				|
| Max pooling	 2x2     	| 10x10x16 in, 5x5x16 out 				|
| Convolution 5x5	    | 5x5x6 in, 1x1x400 out      |
| RELU		|        				| 
| Fully Connected				| Input = 400. Output = 120  |
|RELU					|						|
|Dropout						|				|
|Fully Connected|Input = 120. Output = 84|
| RELU ||
| Dropout ||
|Fully Connected|Input = 84. Output = 43|
 


####3. Trainig the network.

* To train the model, I used AdamOptimizer and following hyperparameters:
* EPOCHS = 15
* BATCH_SIZE = 100
* rate = 0.001

####4. Approach taken
Initially when I trained this network, I was getting 96% validation accuracy but ~60% test accuracy. I realized that I was running to overfitting problem. 



My final model results were:

* validation set accuracy of 95.7%
* test set accuracy of 93.4%

I used the well known LeNet model without any significat modifications. The LeNet architecture was first introduced by LeCun et al. in 1998. It's a very good architecture for recognizing hand writter numbers and it also works well for recognizing images. 

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Five German traffic signs found on the web 
Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

On the notebook, I used 10 signs from the web; however, I am listing the first five signs here. 

I was expecting that sign #1 and #4 may be deficult to identify because they look alike and the 32x32 images does not have enought resolution. To my surprise sign #1 - #4 got identified correctly but not sign #5. However, when I looked at the dataset, I understood why sign #5 did not get correctly identified. We only have 300 images for Roundabout sign. It's a very small data set to train on.

Sign #5 is "Roundabout Mandatory". 

####2. Discuss of the model's predictions 
Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No Passing      		| No Passing  									| 
| No Entry     			| No Entry 										|
| Stop					| Stop											|
| 60 km/h	      		| 60 km/h	  			 				|
| Roundabout| End of no passing by vehicles over 3.5 metric tons	|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 93.4%. 



####3. Model's prediction of the 5 images

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

For all the images model generated 1.0 probabily.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| No Passing  | 


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| No Entry  | 


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Stop  | 


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| 60 km/h  | 


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| End of no passing by vehicles over 3.5 metric tons | 
						




