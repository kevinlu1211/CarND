**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[image1]: ./README_images/car_noncar.png
[image2]: ./README_images/hog.png
[image3]: ./README_images/lab_bb.png
[image4]: ./README_images/hsv_bb.png
[image5]: ./README_images/lab_hm.png
[image6]: ./README_images/lab_bb.png
[image7]: ./README_images/5_video_frames.png


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the cell 15 of the `Vehicle-Detection.ipynb`

I started by reading in all the `vehicle` and `non-vehicle` images.  

#### Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]


#### 2. Explain how you settled on your final choice of HOG parameters.

I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  The color spaces that I explored were the RGB color space, HSV color space and the LAB color space. Overall, the RGB color space performed the worse, which makes sense as unlike the HSV and LAB color spaces the RGB color space isn't robust to changes in lightining. Moreover, the LAB color space seemed to be more immune to noise (when also using the histogram and spatial features) therefore I chose to use the LAB color space at the end.

The parameters were chosen through comparing the validation set accuracy, although an emphasis was also put on the number of features that were needed to build the model; this is because the more `orientations`,`pixels_per_cell` or `cell_per_block` one uses, the more features one would have. Therefore, it would take a longer time to train the model, and make the prediction. After testing various combinations, I found that using `orientations = 9`, `pixels_per_cell = 8` and `cell_per_block = 2` yielded the perfect balance between number of features and accuracy.

#### Here is an example of the HOG features on the A channel of the LAB color space:

![alt text][image2]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using using the HOG features, spatial features, and the color histogram features from the LAB color space. This was done in cell 24.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search was implemented in cell 27 of the notebook. The `scale` was a hard thing to choose because as the scale decreases, the size of the bounding box decreases, and vice-versa, and the more scales more the longer it would take to find the bounding boxes for the cars. I decided on the `scale` and `cells_per_step` (which determines how much the windows overlap) by plotting out many images from the video and running the different scales on short segments of the video. In the end I decided on using scales of `0.9, 1.1, 1.3, 1.5, 1.7` and using `cells_per_step` equal to 2.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on five scales using LAB HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:


#### Here is an example of the bounding boxes found by the sliding window search on the LAB color space:

![alt text][image3]

#### Here is an example of the bounding boxes found by the sliding window search on the HSV color space:

![alt text][image4]


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://www.youtube.com/edit?o=U&video_id=0c9tmgVvgxw) note that this video combines the lane detection algorithm, the code for that can be found [here](https://github.com/kevinlu1211/CarND/tree/master/Part4-Advanced-Lane-Lines)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the unthresholded heat maps for the past 10 frame, and then I stacked the heatmaps on top of each other. I then created bounding boxes by using `scipy.ndimage.measurements.label()`, and also by thresholding the stacked map to identify vehicles. By stacking the heatmaps and also thresholding them I was able to get rid of most of the false positives.

Two methods were attempted to combine the overlapping bounding boxes. The first was assuming each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

#### Here is an example of using the first approach:

![alt text][image5]

The second approach that I used was to use Non-Maximum Suppression, although this didn't work that well

#### Here is an example of using the second approach:

![alt text][image6]

It is evident that the boxes aren't merging that well, and regardless of what parameter I chose for the ratio of overlapping, it didn't solve the problem, although I believe this is due to how the bounding boxes were found. Since the bounding boxes are always squares (due to the implementation), there isn't an overlapping rectangle.


#### Here are five frames and their corresponding heat maps using the first implementation:

![alt text][image7]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main issue that I faced in my implementation of this project wqas trying to find the correct scales and parameters used to produce the bounding boxes. As the majority of the predictive power of my model comes from the HOG features, if there are cases where the lane lines look like the HOG features of the car it would fail there, creating a lot of false positives. 

I believe that this pipeline could be made more robust by using Udacity's images which has the bounding boxes for cars, and using a CNN to detect, and predict the bounding boxes for the cars. Moreover, since the bounding boxes are not restricted to squares, I believe that NMS would work better. Here is a [paper about face alignment using Multi-Task CNNs](https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf)that I have been reading, and I believe that if I re-train the classification and bounding box regression layers of the network then I would be able to use to for the new and improved version of this pipeline.
