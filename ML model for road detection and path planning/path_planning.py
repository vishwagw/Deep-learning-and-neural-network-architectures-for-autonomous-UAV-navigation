import cv2
import numpy as np

def find_target_waypoint(road_mask):
    """Convert road mask to target waypoint"""
    # Skeletonize road
    skeleton = cv2.ximgproc.thinning(road_mask)
    
    # Find centerline points
    points = np.column_stack(np.where(skeleton > 0))
    
    # Select point 20m ahead (assuming 10px = 1m)
    target_idx = min(len(points)-1, 200)  # 20m * 10px/m
    return points[target_idx][::-1]  # Convert to (x,y)

def generate_control(drone_position, target_position):
    """PID controller for drone navigation"""
    Kp_yaw = 0.8
    Kd_yaw = 0.2
    lateral_error = drone_position[0] - target_position[0]
    
    # Yaw control (steering)
    yaw_rate = Kp_yaw * lateral_error 
    
    return {
        'pitch': 0.5,  # Constant forward velocity
        'yaw': yaw_rate,
        'throttle': 0.7  # Altitude hold
    }