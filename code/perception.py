import numpy as np
import cv2
# Helper functions from the lane detection project, SDCND.
import math
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 255, 255], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_line_image(img, lines):
    """
    `img` should be the output of a Canny transform. Lines are the hough lines in that image.
        
    Returns an image with hough lines drawn.
    """
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

######################################################################################

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

def sample_rock_selection_hsv(img):
    # Convert RGB to HSV
    hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # define range of yellow color in HSV
    lower_yellow = np.array([20,100,100])
    upper_yellow = np.array([40,255,255])

    # Threshold the HSV image to get only yellow colors and return the mask
    return cv2.inRange(hsv_image, lower_yellow, upper_yellow)

def obstacle_selection(img, rgb_upper=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    
    # we define "black" as "any of the color channels is above the threshold" 
    thr_black = 0
    is_not_black = (img[:,:,0] > thr_black) | (img[:,:,1] > thr_black) | (img[:,:,2] > thr_black)
    
    # for dark walls, this is not the inverse of the navigable terrain, so there is a gap in the rover vision image
    is_dark = (img[:,:,0] < rgb_upper[0]) & (img[:,:,1] < rgb_upper[1]) & (img[:,:,2] < rgb_upper[2])
            
    # Index the array of zeros with the boolean array and set to 1
    is_wall = is_not_black & is_dark
    color_select[is_wall] = 1
    return color_select

# Define a function to convert to rover-centric coordinates
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = np.absolute(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[0]).astype(np.float)
    return x_pixel, y_pixel

# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to apply a rotation to pixel positions
def rotate_pix(xpix, ypix, yaw):
    # yaw angle is recorded in degrees so first convert to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = xpix * np.cos(yaw_rad) - ypix * np.sin(yaw_rad)
    ypix_rotated = xpix * np.sin(yaw_rad) + ypix * np.cos(yaw_rad)
    return xpix_rotated, ypix_rotated

# Define a function to perform a translation
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    xpix_translated = np.int_(xpos + (xpix_rot / scale))
    ypix_translated = np.int_(ypos + (ypix_rot / scale)) 
    return xpix_translated, ypix_translated

# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image 
    return warped

def calibrate(image):
    # Define calibration box in source (actual) and destination (desired) coordinates
    # These source and destination points are defined to warp the image
    # to a grid where each 10x10 pixel square represents 1 square meter
    # The destination box will be 2*dst_size on each side
    dst_size = 5 
    # Set a bottom offset to account for the fact that the bottom of the image 
    # is not the position of the rover but a bit in front of it
    bottom_offset = 6
    src = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    dst = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  ])
    return src, dst

# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover): 
    image = Rover.img
    
    # 1) Define source and destination points for perspective transform
    source, destination = calibrate(image)
    
    # 2) Apply perspective transform
    warped = perspect_transform(image, source, destination)
    
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    navigable = color_thresh(warped)
    # Hints and Suggestion:
    # For obstacles you can just invert your color selection that you used to detect ground pixels
    # If you've decided that everything above the threshold is navigable terrain,
    # then everthing below the threshold must be an obstacle!
    obstacle = obstacle_selection(warped)
    sample_rock = sample_rock_selection_hsv(warped)   
    
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    vision_im = np.zeros((160, 320, 3), dtype=np.uint8)
    vision_im[:,:,0] = obstacle*255
    vision_im[:,:,1] = sample_rock
    vision_im[:,:,2] = navigable*255
    
    # OPTIONAL Hough line detection
    # 5 by 5 Gaussian Blur
    #blurred = gaussian_blur(navigable, 5)
    # edge detection
    #edges = canny(blurred, 0, 1)
    # apply region of interest mask
    #region = np.array([[[152, 150], [158,150], [320,0], [0,0]]], dtype=np.int32)
    #edges_masked = region_of_interest(edges, region)
    # Hough lines
    #threshold = 40
    #min_line_len = 30
    #max_line_gap = 8
    #rho = 1
    #theta = np.pi/180
    #Rover.hough_lines = cv2.HoughLinesP(edges_masked, rho, theta, threshold, np.array([]), min_line_len, max_line_gap)
    #hough_img = hough_line_image(edges_masked, Rover.hough_lines)
    # blend hough lines over vision image
    #Rover.vision_image = cv2.addWeighted(vision_im,0.7,hough_img,0.3,0)
    
    # here we use the vision image only
    Rover.vision_image = vision_im
    
    # 5) Convert map image pixel values to rover-centric coords
    nav_x, nav_y = rover_coords(navigable)
    rock_x, rock_y = rover_coords(sample_rock)
    
    #Optimizing Map Fidelity Tip:
    #Your perspective transform is technically only valid when roll and pitch angles are near zero.
    #If you're slamming on the brakes or turning hard they can depart significantly from zero, and your transformed
    #image will no longer be a valid map. Think about setting thresholds near zero in roll and pitch to determine which
    #transformed images are valid for mapping.
    is_roll_small = (Rover.roll < 1.00) or (Rover.roll > 359.00) 
    is_pitch_small = (Rover.pitch < 1.00) or (Rover.pitch > 359.00) 
    if is_roll_small and is_pitch_small:
        # 5) Convert map image pixel values to rover-centric coords
        obs_x, obs_y = rover_coords(obstacle)
        # 6) Convert rover-centric pixel values to world coordinates
        rover_xpos = Rover.pos[0]
        rover_ypos = Rover.pos[1]
        rover_yaw = Rover.yaw
        nav_x_world, nav_y_world = pix_to_world(nav_x, nav_y, rover_xpos, rover_ypos, rover_yaw, Rover.worldmap.shape[0], 10)
        obs_x_world, obs_y_world = pix_to_world(obs_x, obs_y, rover_xpos, rover_ypos, rover_yaw, Rover.worldmap.shape[0], 10)
        rock_x_world, rock_y_world = pix_to_world(rock_x, rock_y, rover_xpos, rover_ypos, rover_yaw, Rover.worldmap.shape[0], 10)
        # 7) Update Rover worldmap (to be displayed on right side of screen)
        Rover.worldmap[obs_y_world, obs_x_world, 0] += 1
        Rover.worldmap[rock_y_world, rock_x_world] = [255,255,0]
        Rover.worldmap[nav_y_world, nav_x_world, 2] += 2

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover navigable pixel and rock sample distances and angles
    nav_distances, nav_angles = to_polar_coords(nav_x, nav_y)
    sample_distances, sample_angles = to_polar_coords(rock_x, rock_y)
    Rover.nav_dists = nav_distances
    Rover.nav_angles = nav_angles
    Rover.sample_dists = sample_distances
    Rover.sample_angles = sample_angles
    return Rover