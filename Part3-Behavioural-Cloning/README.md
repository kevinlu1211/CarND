# **Behavioral Cloning** 



**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json
```

#### 3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model arcthiecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 24 and 64 (model.py lines 466-483) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the before being passed into the network (model.py line 506) 

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer after the flattening layer in order to reduce overfitting (model.py line 465). 

The model was trained and validated on different data sets to ensure that the model was not overfitting the data (model.py line 543). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, with the default parameters.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the Udacity data, but created a function to generate more data artifically. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to follow the [Nvidia model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) as this is a tried and tested architecture.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding a dropout layer after the Flatten layer, which improved the test error a bit.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track and I really had no idea what was happening as it seemed that I wasn't overfitting and the test error seemed pretty low. I found out the problem by going back to plotting the images out, and realising that the Udacity data had too many images with a steering angle of 0 which made the car bias to driving straight (into the water LOL). I remedied this by creating a few functions that created more training data. Which will be explain below.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

Here is a visualization of the final model architecture

![Model Visualization](https://github.com/kevinlu1211/CarND/tree/master/Part3-Behavioural-Cloning/model.png)

#### 3. Creation of the Training Set & Training Process

This was the hardest part of the project as I realised that it didn't matter which model architecture I used, the training data was the determining factor to whether the car drove into the water. Using just the raw data from Udacity meant that the car would be biased to driving straight, so I had to create artifical data from the original data. 

My first approach was a simple one, it consists of translating the image in the x direction by a random amount and adding/subtracting an angle which was directly proportional to the amount translation, and then adding or subtracting a fixed amount from the angle according to whether the image was from the left or right camera.

The problem with this was that the car would swerve left and right very often and still drive into the water, as a result of the angle being added/subtracted was proportion to the x translation. I fixed this by first ignoring the generated data with a certain probability if the angle was within some range between -0.1 and 0.1, this fixed the problem of the car driving into the water.

The second fix was to use a different way to determine how much angle to add/subtract given some x translation. I eventually settled on using quadratic extrapolation instead of linear extrapolation, this meant that for small to medium x translations, there wouldn't be much change in the angle, but for medium to large x translations there would be a considerable change in the angle. This fixed swerving for the most part and this was my final iteration of my model.

The training process was quite simply and consisted of splitting the newly generated data into a training, validation and test set, which was then fed into the model.
