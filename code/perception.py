import numpy as np
import cv2

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

# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover): 
    image = Rover.img
    
    # 1) Define source and destination points for perspective transform
    # Define calibration box in source (actual) and destination (desired) coordinates
    # These source and destination points are defined to warp the image
    # to a grid where each 10x10 pixel square represents 1 square meter
    # The destination box will be 2*dst_size on each side
    dst_size = 5 
    # Set a bottom offset to account for the fact that the bottom of the image 
    # is not the position of the rover but a bit in front of it
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  ])
    
    # 2) Apply perspective transform
    warped = perspect_transform(image, source, destination)
    
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    navigable = color_thresh(warped)
    # Hints and Suggestion:
    # For obstacles you can just invert your color selection that you used to detect ground pixels
    # If you've decided that everything above the threshold is navigable terrain,
    # then everthing below the threshold must be an obstacle!
    obstacle = obstacle_selection(warped)
    # optional: smoothing of vision image
    #kernel = np.ones((10,10),np.uint8)
    #obstacle = cv2.morphologyEx(obstacle_selection(warped), cv2.MORPH_OPEN, kernel)
    sample_rock = sample_rock_selection_hsv(warped)   
    
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    Rover.vision_image[:,:,0] = obstacle*255
    Rover.vision_image[:,:,1] = sample_rock
    Rover.vision_image[:,:,2] = navigable*255

    # 5) Convert map image pixel values to rover-centric coords
    nav_x, nav_y = rover_coords(navigable)
    
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
        rock_x, rock_y = rover_coords(sample_rock)
        # 6) Convert rover-centric pixel values to world coordinates
        rover_xpos = Rover.pos[0]
        rover_ypos = Rover.pos[1]
        rover_yaw = Rover.yaw
        nav_x_world, nav_y_world = pix_to_world(nav_x, nav_y, rover_xpos, rover_ypos, rover_yaw, Rover.worldmap.shape[0], 10)
        obs_x_world, obs_y_world = pix_to_world(obs_x, obs_y, rover_xpos, rover_ypos, rover_yaw, Rover.worldmap.shape[0], 10)
        rock_x_world, rock_y_world = pix_to_world(rock_x, rock_y, rover_xpos, rover_ypos, rover_yaw, Rover.worldmap.shape[0], 10)
        # 7) Update Rover worldmap (to be displayed on right side of screen)
        Rover.worldmap[obs_y_world, obs_x_world, 0] += 1
        Rover.worldmap[rock_y_world, rock_x_world, 1] += 96
        Rover.worldmap[nav_y_world, nav_x_world, 2] += 1

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
    rover_centric_pixel_distances, rover_centric_angles = to_polar_coords(nav_x, nav_y)
    Rover.nav_dists = rover_centric_pixel_distances
    Rover.nav_angles = rover_centric_angles
    return Rover