## Project: Search and Sample Return


**The goals / steps of this project are the following:**  

**Training / Calibration**  

* Download the simulator and take data in "Training Mode"
* Test out the functions in the Jupyter Notebook provided
* Add functions to detect obstacles and samples of interest (golden rocks)
* Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.
* Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

**Autonomous Navigation / Mapping**

* Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook). 
* Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands. 
* Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating and mapping.  

[//]: # (Image References)

[image1]: ./successful_run_wall_crawler_1280_600_simple_fps_5.png


## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.
#### Color Selection
OpenCV2 library used.

For selecting the golden rock samples I converted the image to HSV color space. For selecting the thresholds I followed the recommendation of the OpenCV2 documentation. So I chose h=20/40 for the lower/upper hue and 100/255 for the lower/bounds of the other channels. For selecting obstacles I applied thresholds directly to the channels, lower bound is any channel larger than zero (=not black), upper bound all channels below the given threshold of 160.

Additionally I investigated a more logical approach, following the tip from TA. Defined a hlper function that selects non-black pixels. Logic is then `obstacle = not black and not navigable` using the previously masked navigable terrain image.

Possible improvements are to define additional object classes like dark stones and sky, optimize threshold values (the h=20 lower bound for samples and the 160 fro navigable terrain), and make more use of the OpenCV2 library  for more expressive (shorter) code.

#### Additional Investigation: Line detection
Goal is to separate navigable area from obstacles, especially the *dark stones in the middle of the navigable area*. Following the first project of the Self-driving Car Nanodegree, and using some of the helper functions from there, I removed noise with Gaussian blur (kernel size 5) from the `navigable_threshed` image, then detected edges between navigable area and obstacles, then used Hough transformation to detect straight lines in the area of interest (defined polygon to avoid detection of line at edge of camera image). Then displayed the lines in the Rover vision image.

Approach seems viable but slowed down the simulation noticeably. Therefore commented it out in perception.py, but is functional when activated. Still to do is the decision logic for selecting lines indicating dangerous stones ahead.

#### Coordinate Transformations
Followed the exercises from the lessons here. Transformation to polar coordinates will be used to define criteria for decision tree based on distance and angle of navigable area in Rover vision image. Additionally determined minimum and maximum values of navigable area angles, since they can be used to define condition for free terrain (description see below).

#### 2. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result.

#### Image Processing Pipeline for Mapping
1. Read in data. Created workaround helper function for index out of bounds error.
2. Perspective transform from camera perspective to world perspective. Followed the approach from lessons here.
3. Create image masks for navigable terrain, obstacles and rock samples by applying color threshold functions.
4. Coordinate transformation to Rover centric coordinates and then to world coordinates.
5. Updating the worldmap. Increase the blue channel for navigable area, the red channel for obstacles. Mark sample rock pixels white. Possible improvement is to reduce the noise in the resulting map pixels. Solution might be to make the map coloring logic better, so that successive frames are separated. For example, once pixel is marked as navigable, it can never be changed back to *obstacle* by later iteration (same for a sample rock pixel).
6. Generate video. Lower right frame is `data.worldmap`, blurred to reduce noise. See `/output` folder.

### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.
#### Perception Pipeline
The pipeline in `perception_step` is almost the same as the one in the test notebook described above. 

Additionally:

1. Created function for calibration to set source and destination for warping.
2. Added optional code for Hough line detection and showing the Hough lines in the Rover vision image (code works but is commented out, see also explanation above).
3. Implemented the map fidelity tip. The image is considered for mapping only if the pitch and the yaw are deviatiing less than +/-1 degree from 0.0 degrees. This enhances the fidelity a lot.
4. **Created additional fields for the Rover state.** These are `sample_angles` and `sample_dists`. They store the detected rock sample pixels in polar coordinates. They are used in the decision tree for making the Rover aware of a nearby sample. 

#### Decision Logic
The approach is to define a decision tree based on the data coming from the `Rover` object. The data fields used for decision making are current Rover state (`forward` or `stopped`) the angle/distance fields of navigable terrain and rock sample pixels. There are two important functions for evaluating those data:

1. The function `clear_path_ahead` decides if the Rover is able to go forward based on two criteria. Firstly, are there enough free terrain pixels straight ahead? This condition checks if within the angle +/-pi/8 around the forward direction the average pixel distance is more than the threshold `min_free_forward` (set by default to 20). Secondly, does the Rover have enough space to its right and left? Here the aperture angle of the navigable terrain is calculated by subtracting the smallest angle from the largest, and then it is checked if it is larger than `min_aperture`. The path is considered free when both conditions are met. A promising improvement would be to take into account horizontal Hough lines here. They are caused by solid rocks in the middle of the terrain and they would provide a stopping signal (free = straight ahead free and enough space left/right and no horizontal blocking lines).
2. The lambda function `sample_in_sight` is used for deciding whether to approach a sample rock and simply checks if there are a minimum number of sample rock pixels in the data.

The decision tree is structured as follows. On the top level, the Rover state is checked.

1. State is `forward`. If the path ahead is clear we can either approach a rock, if one is in sight, or explore the environment. Otherwise without clear path we go to stop mode.
2. Approaching a sample rock. We use the angles/distances of sample rock pixels to estimate whether we are close to the sample. Since the sample appears like an elongated ellipse in the image, we subtract one standard deviation from the mean distance. If we are closer than 20, we stop. If we are near the sample, the `near_sample` flag is on, we go to `stop` mode, else we accelerate again. The purpose of this logic is to approach the rock slowly. We steer directly towards the sample. Possible improvement is to apply a smooth deceleration function for adjusting the speed.
3. Exploring the environment (mode forward, no sample in sight). Action is to accelerate to maximum velocity. Steering angle is determined by taking the wall crawling approach. The angle is a weighted average of the mean angle of navigable terrain and the minimum angle of free terrain. So here we steer slighty to the right wall, with 0.2 weight. Possible improvement here is to improve the logic for the left/right decision. The direction could change depending on path planning/mapping considerations.
4. State is `stop`. If the rover is still moving it stops completely. If Rover is fully stopped, it an either pick it up a sample when near to it or else try to get unstuck. In this case it checks if the path ahead is clear. The criterion here is that `clear_path_ahead(Rover, 25, 0.6)` and that the navigable terrain is in front of the rover, mean angle is less than 25 degrees. This makes sure that the rover is actually rotating away from the obstacle. If path is free, accelerate with steering set to mean angle of free terrain. Else rotate by +/-15 degrees in the direction of the mean angle of free terrain. Possibly improvements are needed for the criterion for the free path ahead and the setting for the steering angle when rotating.


#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**

I used the following simulator settings resolution **1280 x 600**, quality **simple**. The FPS rate often started at 10 and slowed down to 6.

The rover was able to navigate the terrain, typically with a fidelity of about 70%. Covering enough map area took quite long, path planning will probably improve the performance in this metric. The rover was able to keep a wall to its right, as intended in the current implementation. It was also in some cases able to pick up some of the rocks. 

![Example run.][image1]

The following unwanted behaviour occured and how it might be solved:

- In the open white area the rover gets sometimes stuck in a circle loop. Possible solution is path planning, see below.
- If a sample rock is too close to the walls, the rover aborts the approach, tries to get unstuck and continues with exploration. Solution is to put different criteria for getting unstuck when rover is currently approaching a sample.
- Looping the terrain instead of exploring unmapped territory. Possible solution is path planning, see below.


**Improvements**

- Improve criteria for getting unstuck and for clear path ahead. Create more features like Hough lines that indicate blocking stones. Optimizing threshold values for distances and angles, make those values dependent on current Rover state. Ultimately the rover should make decisions based on evaluation of the entire vision image and not just some handcrafted features like angles and lines. Might be done with a convolutional network.   

- Path planning. Keep a memory of mapped area. When at a crossroads, decide to go for the uncharted territory. Return to the starting point when all sample rocks are collected.


