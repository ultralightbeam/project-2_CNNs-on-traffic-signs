# **Traffic Sign Recognition** 

## Writeup


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[image_all]: ./examples/all_signs.png "all_signs"
[image_bias]: ./examples/distribution.png "dist"
[im_before]: ./examples/im_before.png "before"
[im_after]: ./examples/im_after.png "after"
[valid]: ./examples/valid.png "v"
[im1]: ./examples/im1.png "im1"
[im2]: ./examples/im2.png "im2"
[im3]: ./examples/im3.png "im3"
[im4]: ./examples/im4.png "im4"
[im5]: ./examples/im5.png "im5"


#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

[Project jupyter notebook](https://github.com/ultralightbeam/project-2_CNNs-on-traffic-signs/blob/master/Traffic_Sign_Classifier.ipynb) provided in this repo.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used basic numpy functions to compute statistics of data in the notebook:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

We grab the first sample in each class of the training dataset for a quick understanding of data types.

![Sample from each class][image_all]

We notice from this plot that in some cases the classification may be hard due to low-light conditions.

Below is a simple graph showing the percentage of classes in the dataset. This is almost a necessary check of data bias (e.g. underrepresentation/overrepresentation of certain classes) before jumping into serious ML. 

![Is there a bias][image_bias]

It seems like certain classes such as 0 and >40 are underrepresented compared to classes such as 1-4, consistently in the training, validation, and eval datasets. Thus, we can expect better detection performance for classes of 1-4 than 0, for example.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The input image is first converted into grayscale. Then a simple transform of pixel_new = (pixel-128)/128 is operated as a mean-subtraction and normalization procedure. Below is an example of a traffic sign image before and after preprocessing. Note that the offset and squeeze in color range from preprocessing.

![alt text][im_before]

![alt text][im_after]

The normalization is done here for easy porting of standard NN architectures -- because existing NN architectures commonly assume normalization of data, the normalization eliminates the need for tuning hyperparameters from scratch.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My CNN architecture borrows the standard LeNet-5, but with #(output nodes) modified to 43, which is the number of classes in our application, from 10, which is the MNIST parameter.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscaled image                      |
| Convolution 5x5     	| 1x1 stride, outputs 28x28x6                   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6                  |
| Convolution 5x5     	| 1x1 stride, outputs 10x10x16                  |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16                   |
| Fully connected		| 400 -> 120 nodes								|
| RELU                  |                                               |
| Fully connected		| 120 -> 84 nodes								|
| RELU                  |                                               |
| Fully connected		| 84 -> 43 nodes								|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer with learning rate 0.001. The batch_size was set to 8 and the number of epochs was set to 15.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.6%
* validation set accuracy of 94.5% 
* test set accuracy of 92.4%

The LeNet-5 CNN architecture was chosen as the baseline, since it gave high-performance results in MNIST classification, which is a similar image classification problem to what we are dealing with here. To adapt LeNet-5 to a robust traffic sign classifier, modifications included the grayscale preprocessing, changing #(output nodes) to 43, and retuning hyperparameters to hit 93% validation accuracy See plot below of how validation accuracy grows (with some randomness due to gradient descent being stochastic) with epoch number.


![alt text][valid]


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.


Below are the five German traffic signs found on Google Images (rescaled to 32x32).

![alt text][im1] ![alt text][im2] ![alt text][im3] 
![alt text][im4] ![alt text][im5]

We conjecture that our model would classify these five images robustly, as they avoid the known hard classification scenarios, where the images may be taken at night, through haze/fog, or even partially occluded by other objects (e.g. tall cars, birds).

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30)      | Speed limit (30)								| 
| No entry     			| No entry 										|
| Speed limit (70)      | Speed limit (70)								|
| Stop                  | Stop                                          |
| Turn right ahead		| Turn right ahead      						|

Our CNN model was able to hit 100% accuracy for this very small test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Below we show charts of top-5 prediction probabilities, and claim the  test images are classified by our model extremely confidently. These results can also be found in notebook as inline print.

For the first image (speed limit of 30km/h image), the model is very confident of its decision. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| ~1.0         			| Speed limit (30km/h)   							| 
| 4.4E-21               | Speed limit (20km/h) 										|
| 9.9E-29				| Turn right ahead											|
| 1.9E-31	      		| Speed limit (70km/h)					 				|
| 1.6E-35				| Speed limit (50km/h)      							|

It is intuitive that even though the contending candidates have epsilon probabilities, they are mostly speed limit signs.

For the second image (no entry image), the model is very confident of its decision. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| ~1.0         			| No entry   							| 
| 8.6E-22     				| Turn right ahead 										|
| 2.8E-23					| Yield											|
| 1.2E-24	      			| Turn left ahead					 				|
| 9.8E-28				    | Stop      							|


For the third image (speed limit of 70km/h image), the model is very confident of its decision. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9997         			| Speed limit (70km/h)   							| 
| 2.3E-04     				| Speed limit (20km/h) 										|
| 4.0E-13					| Keep left											|
| 4.3E-17	      			| Roundabout mandatory					 				|
| 1.2E-21				    | General caution      							|



For the fourth image (stop image), the model is very confident of its decision. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9997         			| Stop   							| 
| 2.9E-04     				| Speed limit (80km/h) 										|
| 1.2E-05					| Speed limit (60km/h)											|
| 6.0E-07	      			| Turn left ahead					 				|
| 5.6E-07				    | Road work      							|

The next best ones are speed limits. Possibly because the zeros in 80, 60 match the letter 'O' in stop sign? 


For the fifth image (turn right ahead image), the model is very confident of its decision. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| ~1.0         			| Turn right ahead   							| 
| 4.9E-23     				| Speed limit (20km/h) 										|
| 1.5E-26					| Roundabout mandatory											|
| 1.2E-26	      			| Ahead only					 				|
| 2.1E-29				    | Traffic signals      							|

