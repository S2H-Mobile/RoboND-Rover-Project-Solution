# -*- coding: utf-8 -*-
"""Rover Modes.

This module describes the operation modes of the rover. The modes defined here are the building blocks
of the behaviour stack. They can be combined to from emergent complex behaviour. The states
form a hierarchy where at the base level there are the mission objectives like exploring and
harvesting, as well as the more tactical tasks of getting unstuck and returning to the base. The third
layer consists of action states that define more fine grained actions like stop, turn, accelerate, coast.

Mission Objectives:
    - Explore
    - Harvest
    
Supplemenary Modes:
    - Get unstuck
    - Return to Base

Action States:
    - Accelerate/Coast
    - Stop
    - Turn

Todo:
    * implement all the states given above
    * HARVESTING OF SAMPLE ROCKS
    * extract the decision policies in a separate module

.. _Subsumption architecture on Wikipedia:
   https://en.wikipedia.org/wiki/Subsumption_architecture

"""

class RoverMode(object):
    """
    Superclass for all the Rover states, uses the state machine programming pattern.
    """
    
    def run(self, rover):
        """
        Set the rover state like steering angle, acceleration, brakes.
        """
        assert 0, "run not implemented"
        
    def next(self, rover):
        """
        Decide here for the next state, by applying
        a decision rule or policy to the given rover data. 
        """
        assert 0, "next not implemented"
        
def clear_path_ahead(r, min_free_forward=20, min_aperture=1.0):
    import numpy as np
    # check free distance ahead
    # indices of forward angles
    thr = np.pi/8
    indices = np.where((r.nav_angles > -thr) & (r.nav_angles < thr))

    # evaluate the distances
    if len(indices) > 0:
        forward_pixel_distances = r.nav_dists[indices]
        is_forward_free = np.mean(forward_pixel_distances) > min_free_forward
    else:
        is_forward_free = False

    #check aperture of navigable terrain
    if r.nav_angles is not None and len(r.nav_angles) > 0:
        aperture = abs(np.max(r.nav_angles) - np.min(r.nav_angles))
        is_aperture_wide_enough = aperture > min_aperture
    else:
        is_aperture_wide_enough = False
    return is_forward_free and is_aperture_wide_enough

class Stop(RoverMode):
        
    def run(self, rover):
        rover.brake = 0.3
        rover.throttle = 0.0
        rover.steer = 0.0
        return rover
        
    def next(self, rover):
        if abs(rover.vel) < 0.2:
            if clear_path_ahead(rover):
                return None
            else:
                return Unstuck()
        else:
            # Rover is still moving.
            return self
        
class Explore(RoverMode):
        
    def run(self, r):
        import numpy as np
        r.brake = 0.0
        r.throttle = r.throttle_set
        a = r.nav_angles * 180/np.pi
        mean_angle = np.mean(a)
        min_angle = np.min(a) 
        weighted_angle = 0.2 * (4 * mean_angle + min_angle)
        r.steer = np.clip(weighted_angle, -15, 15)
        return r
        
    def next(self, r):
        if clear_path_ahead(r):
            return self
        else:
            return Stop()
        
class Unstuck(RoverMode):
    
    def run(self, r):
        r.brake = 0.0
        r.throttle = 0.0
        r.steer = 15
        return r
    
    def next(self, r):
        if clear_path_ahead(r, 10, 1.0):
            return None
        else:
            return self