# Implement conditionals to decide what to do given perception data
# Here you're all set up with some basic functionality but you'll need to
# improve on this decision tree to do a good job of navigating autonomously!
#
# Optimizing Time Tip: Moving faster and more efficiently will minimize total time.
# Think about allowing for a higher maximum velocity and give your rover the brains to not revisit previously mapped areas.
#
# Optimizing for Finding All Rocks Tip:
# The sample rocks always appear near the walls.
# Think about making your rover a "wall crawler" that explores the environment
# by always keeping a wall on its left or right. If done right, this optimization can help all the aforementioned metrics.
import numpy as np

# criterion for deciding whether navigable terrain is ahead
# more than 50 navigable pixels in the image
clear_path_ahead = lambda r: len(r.nav_angles) >= r.stop_forward

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
            if clear_path_ahead(Rover):
                if sample_in_sight(Rover):
                    # actual distance is closer than the mean
                    estimated_sample_dist = np.mean(Rover.sample_dists) - np.std(Rover.sample_dists)
                    estimated_sample_angle = np.mean(to_deg(Rover.sample_angles))
                    print("Sample in sight at mean distance {} and mean angle {}".format(estimated_sample_dist, estimated_sample_angle))
                    # decide throttle and brake
                    # speed proportional to distance
                    if estimated_sample_dist < 20.0:
                        print("Rock sample in distance {} which is less than 20.0, approaching.".format(estimated_sample_dist))
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
                    # steering
                    steering_angle = np.clip(estimated_sample_angle, -15, 15)
                    print("Set steering angle to {}.".format(steering_angle))
                    Rover.steer = steering_angle
                else:
                    #mode is forward, navigable terrain looks good 
                    # and velocity is below max, then throttle 
                    if Rover.vel < Rover.max_vel:
                        # Set throttle value to throttle setting
                        Rover.throttle = Rover.throttle_set
                    else: # Else coast
                        Rover.throttle = 0
                    # dont brake
                    Rover.brake = 0
                    # Set steering to average angle clipped to the range +/- 15   
                    # here we can apply pid
                    nav_angle = np.mean(to_deg(Rover.nav_angles))
                    if Rover.hough_lines is not None:
                        theta = 0
                        for x1,y1,x2,y2 in Rover.hough_lines[0]:
                            theta = to_deg(np.arctan2(y2-y1, x2-x1))
                            print("nav_angle {}, theta {}".format(nav_angle,theta))
                        raw_angle = 0.5 * (nav_angle + theta)
                        print("raw angle {}".format(raw_angle))
                    else:
                        raw_angle = nav_angle
                    steering_angle = np.clip(raw_angle, -15, 15)
                    print("Set steering angle to {}.".format(steering_angle))
                    Rover.steer = steering_angle
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
                    if len(Rover.nav_angles) < Rover.go_forward:
                        Rover.throttle = 0
                        # Release the brake to allow turning
                        Rover.brake = 0
                        # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                        Rover.steer = -15 # Could be more clever here about which way to turn
                    # If we're stopped but see sufficient navigable terrain in front then go!
                    if len(Rover.nav_angles) >= Rover.go_forward:
                        # Set throttle back to stored value
                        Rover.throttle = Rover.throttle_set
                        # Release the brake
                        Rover.brake = 0
                        # Set steer to mean angle
                        Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                        Rover.mode = 'forward'
    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        print("No Rover.nav_angles data available.")
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0

    return Rover

