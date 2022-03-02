import pdb
import os
import cv2
import numpy as np
import pandas as pd
from cv2 import STEREO_BM_PREFILTER_NORMALIZED_RESPONSE

ENCODER_PATH = 'data/sensor_data/encoder.csv'
FOG_PATH = 'data/sensor_data/fog.csv'
LIDAR_PATH = 'data/sensor_data/lidar.csv'



#--------------Parameters we will use ------------------

ENCODER_PARAMETER = {
    'resolution': 4096,
    'left_wheel_diameter': 0.623479,
    'right_wheel_diameter': 0.622806,
    'wheel_base': 1.52439
}

LIDAR_PARAMETER = {
    'fov': 190, 
    'start_angle': -5, 
    'end_angle': 185, 
    'angular_resolution': 0.666,
    'max_range': 80,
    'RPY': [142.759, 0.0584636, 89.9254],
    'R': np.array([[0.00130201, 0.796097, 0.605167], 
                [0.999999, -0.000419027, -0.00160026], 
                [-0.00102038, 0.605169, -0.796097]]),
    'T': np.array([0.8349, -0.0126869, 1.76416])
}

STEREO_PARAMETER = {
    'baseline':  475.143600050775   #(mm) 
}

VEHICLE2FOG = {
    'RPY': [0, 0, 0],
    'R': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    'T': np.array([[-0.335], [-0.035], [0.78]])
}

VEHICLE2LIDAR = {
    'RPY': [142.759, 0.0584636, 89.9254],
    'R': np.array([[0.00130201, 0.796097, 0.605167], 
                [0.999999, -0.000419027, -0.00160026], 
                [-0.00102038, 0.605169, -0.796097]]),
    'T': np.array([0.8349, -0.0126869, 1.76416])
}

VEHICLE2STEREO = {
    'baseline': 475.143600050775,
    'RPY': [-90.878, 0.0132, -90.3899],
    'R': np.array([[-0.00680499, -0.0153215, 0.99985   ],
                    [-0.999977, 0.000334627, -0.00680066],
                    [-0.000230383, -0.999883, -0.0153234]]),
    'T': np.array([1.64239, 0.247401, 1.58411]),
    'left_matrix': np.array([[8.1690378992770002e+02, 5.0510166700000003e-01, 6.0850726281690004e+02],
                            [0., 8.1156803828490001e+02, 2.6347599764440002e+02], 
                            [0., 0., 1. ]]),
    'right_matrix': np.array([[8.1378205539589999e+02, 3.4880336220000002e-01, 6.1386419539320002e+02], 
                            [0., 8.0852165574269998e+02, 2.4941049348650000e+02], 
                            [0., 0., 1. ]]),
    'projection_matrix': np.array([[7.7537235550066748e+02, 0., 6.1947309112548828e+02, -3.6841758740842312e+02],
                                [0., 7.7537235550066748e+02, 2.5718049049377441e+02, 0.], 
                                [0., 0., 1., 0. ]]),
    'image_width': 1280,
    'image_height': 560
}

STEREO = {
    'STEREO_POSITION': np.array([1.64239, 0.247401, 1.58411]),
    'STEREO_LEFT_CAMERA': np.array([[ 8.1690378992770002e+02, 5.0510166700000003e-01,
        6.0850726281690004e+02],
        [0., 8.1156803828490001e+02,
        2.6347599764440002e+02], [0., 0., 1. ]]),
    'STEREO_IMG_WIDTH': 1280,
    'STEREO_IMG_HEIGHT': 560,
    'STEREO_Z_RANGE': [-0.3, 0.3],
    'STEREO_BASELINE': 0.475143600050775
}

def read_data_from_csv(filename):
    '''
    INPUT 
    filename        file address

    OUTPUT 
    timestamp       timestamp of each observation
    data            a numpy array containing a sensor measurement in each row
    '''
    data_csv = pd.read_csv(filename, header=None)
    data = data_csv.values[:, 1:]
    timestamp = data_csv.values[:, 0]
    return timestamp, data


def read_encoder(path):
    """
    get velocity from encoder data
    """
    time_stamp, data = read_data_from_csv(path)
    dt = time_stamp[1:] - time_stamp[:-1]
    velocity = (data[1:] - data[:-1]) / dt[:, None]
    velocity = velocity / ENCODER_PARAMETER['resolution']
    velocity = velocity * np.array([ENCODER_PARAMETER['left_wheel_diameter'], ENCODER_PARAMETER['right_wheel_diameter']]) * np.pi
    return time_stamp[:-1], np.mean(velocity, axis=1)

def read_fog(path):
    time_stamp, data = read_data_from_csv(path)
    dt = time_stamp[1:] - time_stamp[:-1]
    yaw_speed = data[1:, 2] / dt
    return time_stamp[:-1], yaw_speed

def read_lidar(path):
    time_stamp, data = read_data_from_csv(path)
    return time_stamp[:-1], data

def read_disparity():
    if os.path.exists("data/disparity.npy") and os.path.exists("data/time.npy"):
        time_stamp, disparity = np.load("data/time.npy"), np.load("data/disparity.npy")
        # print(type(time_stamp))
        # print(time_stamp.shape, disparity.shape)
        # pdb.set_trace()
    else: 
        path_l = 'data/stereo_images/stereo_left'
        path_r = 'data/stereo_images/stereo_right'
        stereo = cv2.StereoBM_create(numDisparities=32, blockSize=9) 
        disparity_list, time_list = [], []
        for filename in os.listdir(path_l):
            time_stamp = os.path.splitext(filename)[0]
            image_l = cv2.imread(os.path.join(path_l, filename), 0)
            image_r = cv2.imread(os.path.join(path_r, filename), 0)
            if image_l is not None and image_r is not None:  # there are some missing...
                image_l = cv2.cvtColor(image_l, cv2.COLOR_BAYER_BG2BGR)
                image_r = cv2.cvtColor(image_r, cv2.COLOR_BAYER_BG2BGR)
                image_l_gray = cv2.cvtColor(image_l, cv2.COLOR_BGR2GRAY)
                image_r_gray = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY)
                disparity = stereo.compute(image_l_gray, image_r_gray)
                disparity[disparity < 0] = 0
                disparity_list.append(disparity)
                time_list.append(time_stamp)
        time_stamp, disparity = np.array(time_list).astype(np.int64), np.array(disparity_list)
        np.save("data/disparity.npy", disparity)
        np.save("data/time.npy", time_stamp)
    return time_stamp, disparity

# Stereo camera (based on left camera) extrinsic calibration parameter from vehicle
# RPY(roll/pitch/yaw = XYZ extrinsic, degree), R(rotation matrix), T(translation matrix)
# RPY: -90.878 0.0132 -90.3899
# R: -0.00680499 -0.0153215 0.99985 -0.999977 0.000334627 -0.00680066 -0.000230383 -0.999883 -0.0153234 
# T: 1.64239 0.247401 1.58411

"""
Lidar sensor (LMS511) extrinsic calibration parameter from vehicle
RPY(roll/pitch/yaw = XYZ extrinsic, degree), R(rotation matrix), T(translation matrix)
RPY: 142.759 0.0584636 89.9254
R: 0.00130201 0.796097 0.605167 0.999999 -0.000419027 -0.00160026 -0.00102038 0.605169 -0.796097 
T: 0.8349 -0.0126869 1.76416
"""

# * LiDAR rays with value 0.0 represent infinite range observations.
# with open(left_camera, 'r') as f:
#     x = yaml.load(f, Loader=yaml.FullLoader)

if __name__ == '__main__':
    read_disparity()
