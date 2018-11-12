# **Advanced Lane Finding**

## Kavan Brandon

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
"camera_cal/calibration_pickle.p"
[image1]: ./camera_cal/corners_found3.jpg "Distorted"
[image2]: ./camera_cal/corners_found4_undistorted.jpg "Undistorted Calibration"
[image3]: ./output_images/undistorted_image_output.jpg "Undistorted Lane Image"
[image4]: ./output_images/sobel_output.jpg "Sobel Gradient"
[image5]: ./output_images/color_threshold_and_sobel.jpg "Combined Sobel and Color Binary"
[image6]: ./output_images/warped_output.jpg "Warped Image"
[image7]: ./output_images/lane_line_identification.jpg "Lane Line Identification"
[image8]: ./output_images/final_result.jpg "Final Result"
[video1]: ./project_video.mp4 "Video"

### Required Files

#### 1. Project files

My project includes the following files:
* video_gen.py for creating an video pipeline
* Advanced_Lane_Line_Marker_Pipeline.py for camera calibration
* image_gen.py for creating an image pipeline
* writeup_report.md summarizing the results
* output1_tracked.mp4 for showing the recorded drive in autonomous mode
* camera_cal folder for storing distortion coefficients and calibration images
* output_images folder with example pipeline image outputs

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

In order to compute the camera matrix and distortion coefficients, it was first necessary to find the chessboard corners. As explained the project description, I set the number of columns to 9 and the number of rows to 6. I prepared individual object points by using the numpy zeros function to ensure each point holds x, y, and z values. Furthermore, `objpoints` and `imgpoints` arrays were created to store 3D real world space points and 2D image plane points respectively.

The next step required iterating through all of the calibration images in order to find chessboard corners and draw the found corners on each image. Beforehand, all images were converted to grayscale. I used the cv2 `findChessboardCorners` to find the chessboard corners for each image. If the function properly returned the corners, I proceeded to draw on each corner using the cv2 `drawChessboardCorners` function. Once all `objpoints` and `imgpoints` arrays were populated, I calculated the calibration and distortion coefficients using the cv2 `calibrateCamera` function. These coefficients were then saved to a pickle file.

The below image shows an distorted chessboard image with drawn corners:

![alt text][image1]

The below image shows the undistorted chessboard image with drawn corners:

![alt text][image2]

Several images were unable to be calibrated because all chessboard corners were not exposed by the image. Since the function could not find the appropriate number of columns and rows, the cv2 framework was not able to detect all possible corners.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Here is an example of a distorted corrected image:
![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I first experimented using a sobel threshold on the x and y derivatives (lines 109-110 in image_gen.py). As described in the lesson, the sobel operator is an important algorithm used for Canny edge detection. Here is an example of the output:

![alt text][image4]

Next, I combined the sobel threshold with color thresholding(line 111 in image_gen.py). The color thresholding uses a combination of HLS and HSV conversions to isolate the V and S channels and apply them to the final output. This seemed necessary for removing low saturation and low pixel values in order to accentuate the lane lines. This was the final output of both color and sobel thresholding:

![alt text][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for the perspective transform can be found from lines 116 to 133 in image_gen.py. I hardcoded the source and destination numbers using the same values found in the project Q&A discussion. They seemed to work relatively well. However, my code was slightly different for organizing the numpy array containing the source and destination points:

```python
  bot_width = .76
  mid_width = .08
  height_pct = .62
  bottom_trim = .935

  top_left =  (img.shape[1]*(0.5-mid_width/2), img.shape[0]*height_pct)
  top_right = (img.shape[1]*(0.5+mid_width/2), img.shape[0]*height_pct)
  bottom_left =  (img.shape[1]*(0.5-bot_width/2), img.shape[0]*bottom_trim)
  bottom_right = (img.shape[1]*(0.5+bot_width/2), img.shape[0]*bottom_trim)

  src = np.float32([top_left, top_right, bottom_left, bottom_right])

  offset = img_size[0]*.25
  dst = np.float32([[img.shape[1]*.25,0],[img.shape[1]*.75, 0], [img.shape[1]*.25, img.shape[0]], [img.shape[1]*.75, img.shape[0]]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 589, 446      | 320, 0        |
| 154, 673      | 320, 720      |
| 1126, 673     | 960, 720      |
| 691, 446      | 960, 0        |

Here is an example of a warped image:

![alt text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code where lane-line pixels are identified and fit with polynomials can be found on lines 138-163 in image_gen.py. Using a sliding window approach, the goal was fitting polynomials where pixel values identified lane lines.

Here is a visualization of lane line identification:

![alt text][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of the curve and the position of the vehicle with respect to the center was calculated in lines 198 to 108 in image_gen.py. The calculations involve calculating the radius of curvature based on pixel values, then converting the values to real world space (line 198), then calculating the radius of the curvature based on real world space.

```python
curve_fit_cr = np.polyfit(np.array(res_yvals, np.float32)*ym_per_pix, np.array(leftx, np.float32)*xm_per_pix, 2)
curverad = ((1 + (2*curve_fit_cr[0]*yvals[-1]*ym_per_pix + curve_fit_cr[1])**2)**1.5) / np.absolute(2*curve_fit_cr[0])
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Lines 179-189 in image_gen.py add polyfills that clearly indicate the left and right lane lanes, in addition to filling the lane area between the two lines with a green fill color. Here is the final result on the road image:

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output1_tracked.mp4 )

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

It seems the pipeline would fail on images that include heavily curved roads where the pre-set source and destination points are hard-coded. These pre-set points used during perspective transform seem to work generally on flat surfaces with small to medium sized curves. Furthermore, I don't believe these points would successfully work for hilly roads where lane lines could drop-off ahead of the car. The pipeline could be made more robust using a machine learning model to dynamically detect the source and destination points based on a large dataset of lane images in various weather conditions, lane curvatures, and distances from the exact center of the lane. Lastly, machine learning techniques could be applied to choosing the best gradient and colors depending on the type of weather. Color and gradient decisions may not work best in non-traditional weather conditions including rain, snow, or heavily shadowed areas.
