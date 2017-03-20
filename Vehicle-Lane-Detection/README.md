## Advanced Lane Finding

#### Structure of the code files
The Lane Detection.ipynb provides a better picture of all the things that I have tried in trying to recognize the lane lines, such as using different thresholds for the sobel operator, or using different color spaces and different transforms (such as the tophat transform). Although if those things don't interest you here is a [link to my video result](https://youtu.be/TjinyUHjR_8).

[//]: # (Image References)

[image1]: ./README_images/road_calibration.png "Chessboard Transformed"
[image2]: ./README_images/chessboard_calibration.png "Road Transformed"
[image3]: ./README_images/stacked_image.png "Stacked Image"
[image4]: ./README_images/thresholded_stacked_image.png "Thresholded Stacked Image"
[image5]: ./README_images/perspective_transformed_points.png "Perspective Transformed Thresholded Stacked Image"
[image6]: ./README_images/perspective_transformed_binary.png "Perspetive Transformed Binary Stacked Image"
[image7]: ./README_images/binary_fitted_lanes.png "Binary Fitted Lane Image"
[image8]: ./README_images/filled_lane_line.png "Filled Lane Line"


---
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the code cells 2-15 of the IPython notebook located 
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained these before and after images for the chessboard: 

![alt text][image1]

### Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
In this section I did a lot of experiment with 2 kinds of images the first was using a combination of the RGB, HLS, and LAB image spaces. This can be seen through the code cells 25-45. I used R, G channels from the RGB color space, the L, B channnels from the LAB color space, and the S channel from the HLS color space. Then using these channels, I did various graident and magnitude thresholding methods to each the lane lines for each color channel and then stacked them on top of each other. I obtained this as a result:
![alt text][image3]
Now to get rid of the noise, I thresholded it by setting each pixel in the image had a value of greater than 3, as I had use 6 thresholded images (similar to heatmap thresholding). If the value was greater or equal to 3 then the pixel stays, if not then it is set to 0. I obtained this as a result

![alt text][image4]

The second kind of images that I experimented with were the tophat transforms of the RGB, HLS and, LAB channels. Although I will not go into further detail as they were not used. If you are interested though, thery are in cells 50-75.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in cell 18   The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

The way that I chose the points for the perspective transform was base on an implementation by trying to find the vanishing point of the image, and using that point to derive the source points. [Ajsmiltuin](https://github.com/ajsmilutin/CarND-Advanced-Lane-Lines) has implemented this and also provided a nice explaination on how it works.

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 374, 479      | 0  , 0        | 
| 904, 479      | 1280, 0      |
| 1812, 685     | 1280, 720      |
| -533, 685      | 0, 720|

![alt text][image5]


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I fit a 2 degree polynomial function on the perspective transformed binary image in cells 81 and I got this as a result:

![alt text][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature and the offset from the center of lane were calculated in cells 82-83. The radius of curvature was calculated from the equation taken from [here](http://www.intmath.com/applications-differentiation/8-radius-curvature.php) and the offset was simply calculated by taking the difference in pixel values between the base of the left and right lane lines, then converting that value into meters

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines in cell.  Here is an example of my result on a test image:

![alt text][image8]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main problem by far were the difference in lighting changes, and also when a vehicle passes near the car, the warped binary image will include images of that car and a result the pipeline will fail to find the lane lines. I could make this more robust by implementing a function/or using `HoughLinesP` to detect the lines in the binary transformed image, and getting rid of the other pixels. As for the lighting changes, this is a bit harder to deal with but I believe that with some research I will be able to use a variation of different color spaces to more accurately detect the lane lines in changes in lighting conditions. 


## Vehicle Detection

The goals / steps of this pipeline are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[image9]: ./README_images/car_noncar.png
[image10]: ./README_images/hog.png
[image11]: ./README_images/lab_bb.png
[image12]: ./README_images/hsv_bb.png
[image13]: ./README_images/lab_hm.png
[image14]: ./README_images/lab_nms.png
[image15]: ./README_images/5_video_frames.png


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the cell 15 of the `Vehicle-Detection.ipynb`

I started by reading in all the `vehicle` and `non-vehicle` images.  

#### Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image9]


#### 2. Explain how you settled on your final choice of HOG parameters.

I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  The color spaces that I explored were the RGB color space, HSV color space and the LAB color space. Overall, the RGB color space performed the worse, which makes sense as unlike the HSV and LAB color spaces the RGB color space isn't robust to changes in lightining. Moreover, the LAB color space seemed to be more immune to noise (when also using the histogram and spatial features) therefore I chose to use the LAB color space at the end.

The parameters were chosen through comparing the validation set accuracy, although an emphasis was also put on the number of features that were needed to build the model; this is because the more `orientations`,`pixels_per_cell` or `cell_per_block` one uses, the more features one would have. Therefore, it would take a longer time to train the model, and make the prediction. After testing various combinations, I found that using `orientations = 9`, `pixels_per_cell = 8` and `cell_per_block = 2` yielded the perfect balance between number of features and accuracy.

#### Here is an example of the HOG features on the A channel of the LAB color space:

![alt text][image10]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using using the HOG features, spatial features, and the color histogram features from the LAB color space. This was done in cell 24.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search was implemented in cell 27 of the notebook. The `scale` was a hard thing to choose because as the scale decreases, the size of the bounding box decreases, and vice-versa, and the more scales more the longer it would take to find the bounding boxes for the cars. I decided on the `scale` and `cells_per_step` (which determines how much the windows overlap) by plotting out many images from the video and running the different scales on short segments of the video. In the end I decided on using scales of `0.9, 1.1, 1.3, 1.5, 1.7` and using `cells_per_step` equal to 2.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on five scales using LAB HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  


#### Here is an example of the bounding boxes found by the sliding window search on the LAB color space:

![alt text][image11]

#### Here is an example of the bounding boxes found by the sliding window search on the HSV color space:

![alt text][image12]

As mentioned above, the HSV spaces seems to have more false positives than the LAB space.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://www.youtube.com/watch?v=0c9tmgVvgxw) note that this video combines the lane detection (above) with the vehicle detection algorithm.


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the unthresholded heat maps for the past 10 frame, and then I stacked the heatmaps on top of each other. I then created bounding boxes by using `scipy.ndimage.measurements.label()`, and also by thresholding the stacked map to identify vehicles. By stacking the heatmaps and also thresholding them I was able to get rid of most of the false positives.

Two methods were attempted to combine the overlapping bounding boxes. The first was assuming each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

#### Here is an example of using the first approach:

![alt text][image13]

The second approach that I used was to use Non-Maximum Suppression, although this didn't work that well

#### Here is an example of using the second approach:

![alt text][image14]

It is evident that the boxes aren't merging that well, and regardless of what parameter I chose for the ratio of overlapping, it didn't solve the problem, although I believe this is due to how the bounding boxes were found. Since the bounding boxes are always squares (due to the implementation), there isn't an overlapping rectangle.


#### Here are five frames and their corresponding heat maps using the first implementation:

![alt text][image15]

The left column shows the output of the bounding boxes found without the threshold. The second column is the bounding boxes drawn on the thresholded heatmaps (which gets rid of the false positives). The third column is the thresholded heatmap.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main issue that I faced in my implementation of this project wqas trying to find the correct scales and parameters used to produce the bounding boxes. As the majority of the predictive power of my model comes from the HOG features, if there are cases where the lane lines look like the HOG features of the car it would fail there, creating a lot of false positives. 

I believe that this pipeline could be made more robust by using Udacity's images which has the bounding boxes for cars, and using a CNN to detect, and predict the bounding boxes for the cars. Moreover, since the bounding boxes are not restricted to squares, I believe that NMS would work better. Here is a [paper about face alignment using Multi-Task CNNs](https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf) that I have been reading, and I believe that if I re-train the classification and bounding box regression layers of the network then I would be able to use to for the new and improved version of this pipeline.
