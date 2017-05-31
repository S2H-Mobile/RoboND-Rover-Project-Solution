# Implement conditionals to decide what to do given perception data
# Here you're all set up with some basic functionality but you'll need to
# improve on this decision tree to do a good job of navigating autonomously!
#
# Implemented: Optimizing for Finding All Rocks Tip
# The sample rocks always appear near the walls.
# Think about making your rover a "wall crawler" that explores the environment
# by always keeping a wall on its left or right. If done right, this optimization can help all the aforementioned metrics.
#
# Todo: Optimizing Time Tip:
# Moving faster and more efficiently will minimize total time.
# Think about allowing for a higher maximum velocity and give your rover the brains to not revisit previously mapped areas.
import numpy as np

# criterion for deciding whether navigable terrain is ahead
def clear_path_ahead(r, min_free_forward=20, min_aperture=1.0):
    # check free distance ahead
    # indices of forward angles
    thr = np.pi/8
    indices = np.where((r.nav_angles > -thr) & (r.nav_angles < thr))
    if len(indices) > 0:
        forward_pixel_distances = r.nav_dists[indices]
        is_forward_free = np.mean(forward_pixel_distances) > min_free_forward
    else:
        is_forward_free = False
    print("Is forward direction free? {}".format(is_forward_free))
    
    #check aperture of navigable terrain
    if r.nav_angles is not None and len(r.nav_angles) > 0:
        aperture = abs(np.max(r.nav_angles) - np.min(r.nav_angles))
        print("Aperture {}".format(aperture))
        is_aperture_wide_enough = aperture > min_aperture
    else:
        is_aperture_wide_enough = False
    print("Is aperture wide enough? {}".format(is_aperture_wide_enough))
    
    # consider path clear if both conditions are met
    return is_forward_free and is_aperture_wide_enough

# criterion for deciding whether rock sample is in sight
# more than 4 rock sample pixels in the image 
sample_in_sight = lambda r: len(r.sample_angles) >= 4

# convert from radians to degrees
to_deg = lambda theta_rad: theta_rad * 180/np.pi

# This is the decision tree for determining throttle, brake and steer commands
# based on the output of the perception_step() function.
def decision_step(Rover):
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        # Check for Rover.mode status
        if Rover.mode == 'forward':
            print("Rover in forward mode.")
            # Check the extent of navigable terrain
            if clear_path_ahead(Rover, 20, 1.2):
                if sample_in_sight(Rover):
                    # Approaching rock sample
                    # actual distance is closer than the mean
                    estimated_sample_dist = np.mean(Rover.sample_dists) - np.std(Rover.sample_dists)
                    estimated_sample_angle = np.mean(to_deg(Rover.sample_angles))
                    print("Sample in sight at mean distance {} and mean angle {}".format(estimated_sample_dist, estimated_sample_angle))
                    # decide throttle and brake
                    # speed proportional to distance
                    if estimated_sample_dist < 20.0:
                        print("Detected rock sample in distance {} < 20.0, approaching sample.".format(estimated_sample_dist))
                        if Rover.near_sample:
                            print("Rock sample near, stopping.")
                            Rover.throttle = 0
                            Rover.brake = Rover.brake_set
                            Rover.mode = 'stop'
                        else:
                            print("Rock sample not near, coasting.")
                            Rover.throttle = Rover.throttle_set
                            Rover.brake = 0
                    else:
                        target_speed = np.clip(estimated_sample_dist / 40.0, 0, Rover.max_vel)
                        print("Target speed towards sample is {}, current velocity is {}".format(target_speed, Rover.vel))
                        if Rover.vel < target_speed:
                            print("Accelerate.")
                            Rover.throttle = Rover.throttle_set
                            Rover.brake = 0
                        else:
                            print("Brake.")
                            Rover.throttle = 0
                            Rover.brake = Rover.brake_set
                            
                    # set steering
                    steer_angle_approach_sample = np.clip(estimated_sample_angle, -20, 20)
                    print("Set steering angle to {}.".format(steer_angle_approach_sample))
                    Rover.steer = steer_angle_approach_sample
                else:
                    # Exploring
                    # mode is forward, terrain appears anvigable
                    
                    # set throttle
                    # if velocity is below max, then throttle, else coast 
                    if Rover.vel < Rover.max_vel:
                        Rover.throttle = Rover.throttle_set
                    else:
                        Rover.throttle = 0
                        
                    # set brake: dont brake
                    Rover.brake = 0
                    
                    # Set steering to weighted average of mean angle and right edge of navigable area   
                    # trying to always keep a wall on its right
                    # alternative approach would be to use Hough lines here instead min_angle
                    mean_angle = np.mean(to_deg(Rover.nav_angles))
                    min_angle = np.min(to_deg(Rover.nav_angles)) 
                    weighted_angle = 0.2 * (4 * mean_angle + min_angle)
                    print("Weighted angle {}.".format(weighted_angle))
                    steer_angle_exploring = np.clip(weighted_angle, -15, 15)
                    print("Set steering angle to {}.".format(steer_angle_exploring))
                    Rover.steer = steer_angle_exploring
                    
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            #elif len(Rover.nav_angles) < Rover.stop_forward:
            else:
                # Set mode to "stop" and hit the brakes!
                Rover.throttle = 0
                # Set brake to stored brake value
                Rover.brake = Rover.brake_set
                Rover.steer = 0
                Rover.mode = 'stop'

        # Check for Rover.mode status
        elif Rover.mode == 'stop':
            print("Rover in stop mode.")
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                print("Coming to a halt.")
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                print("Rover has stopped.")
                if Rover.near_sample:
                    print("Rover is near a sample.")
                    if not Rover.picking_up:
                        print("Trigger sample pickup.")
                        Rover.send_pickup = True
                        Rover.throttle = 0
                        Rover.brake = 0
                        Rover.steer = 0
                else:
                    print("Getting unstuck.")
                    # Now we're stopped and we have vision data to see if there's a path forward
                    # If we're stopped but see sufficient navigable terrain in front then go!
                    #if len(Rover.nav_angles) >= Rover.go_forward:
                    if clear_path_ahead(Rover, 16, 0.5):
                        # Set throttle back to stored value
                        Rover.throttle = Rover.throttle_set
                        # Release the brake
                        Rover.brake = 0
                        # Set steer to mean angle
                        Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                        Rover.mode = 'forward'
                        print("Terrain seems free, accelerating.")
                    else:
                        Rover.throttle = 0
                        # Release the brake to allow turning
                        Rover.brake = 0
                        # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                        # turn to the left, since we keep rocks on the right during exploration
                        if np.mean(Rover.nav_angles) > 0:
                            steer = 15
                        else:
                            steer = -15
                        Rover.steer = steer
                        print("Terrain still blocked, steer {}.".format(steer))
    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        print("No Rover.nav_angles data available.")
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0

    return Rover

