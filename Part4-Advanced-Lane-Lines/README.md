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

#### 1. Provide an example of a distortion-corrected image.
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

