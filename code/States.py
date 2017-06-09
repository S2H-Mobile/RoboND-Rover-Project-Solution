class RoverMode(object):
    def run(self, rover):
        assert 0, "run not implemented"
    def next(self, rover):
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