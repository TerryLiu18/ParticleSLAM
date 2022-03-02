import os
import cv2
import argparse
import numpy as np
from map import Map
from math import *
from particles import Particles
from matplotlib import pyplot as plt
from pr2_utils import *
from data import (
    ENCODER_PATH, ENCODER_PARAMETER, FOG_PATH, VEHICLE2FOG,
    LIDAR_PATH, LIDAR_PARAMETER, VEHICLE2LIDAR, STEREO, 
    VEHICLE2FOG, VEHICLE2STEREO, read_data_from_csv,
    read_encoder, read_fog, read_lidar, read_disparity  
)
from tqdm import tqdm

map = Map((-100, 1500), (-1500, 100), 1)
particles = Particles(map=map, n_particles=20, n_efficient_threshold=0.1)

encoder_time, encoder_data = read_encoder(ENCODER_PATH)
yaw_time, yaw_data = read_fog(FOG_PATH)
lidar_time, lidar_points = read_data_from_csv(LIDAR_PATH)
lidar_points = lidar_check(lidar_points)
stereo_time, disparity = read_disparity()

if not os.path.exists("output"):
    os.mkdir("output")
    print('setting output directory')


camera_T = calculate_camera()
car_event = {}
color_map = np.zeros((3, *map.map_value.shape), dtype=np.uint8)

def _append_record(time, event):
    for idx, t in enumerate(time):
        if t in car_event:
            car_event[t].append((event, idx))
        else:
            car_event[t] = [(event, idx)]

_append_record(encoder_time, "encoder")
_append_record(yaw_time, "yaw")
_append_record(lidar_time, "lidar")
_append_record(stereo_time, "stereo")
timestamp = sorted(car_event.keys())

last_t, velocity, angle_velocity = None, None, None
trace_x, trace_y = [], []


counter = 0
for t in tqdm(timestamp):
    events = car_event[t]
    counter += 1

    # there can be multiple events at the same time
    for event in events:
        data_type, idx = event
        if data_type == "encoder": # velocity
            velocity = encoder_data[idx]
            if last_t is None:
                if (velocity is not None) and (angle_velocity is not None):
                    last_t = t
            else:
                particles.predict(velocity, angle_velocity, t-last_t)
                last_t = t

        elif data_type == "yaw":  # angular velocity
            angle_velocity = yaw_data[idx]
            if last_t is None:
                if (velocity is not None) and (angle_velocity is not None):
                    last_t = t
            else:
                particles.predict(velocity, angle_velocity, t-last_t)
                last_t = t

        elif data_type == "lidar":
            max_particle = np.argmax(particles.weights)
            lidar_point = lidar_points[idx, :]
            particles.update_weights(lidar_point)
            particles.update_map(max_particle, lidar_point)
            particles.resample()

        # online texture mapping
        elif data_type == "stereo":
            # Read RGB image
            path = os.path.join('data/stereo_images/stereo_left', "{}.png".format(stereo_time[idx]))
            img = cv2.imread(path, 0)
            img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            disparity_map = disparity[idx, :, :]
            camera_co = get_world_p(camera_T, disparity_map)
            valid_idx = (disparity_map > 0).reshape(-1)
            target_T = camera_co.reshape((3, -1))[:, valid_idx]
            img = (img.reshape((-1, 3)).T)[:, valid_idx]
            max_particle = np.argmax(particles.weights)

            position = np.concatenate([particles.position[max_particle, :], np.zeros(1)])
            rotation = np.array([[cos(particles.orientation[max_particle]), -sin(particles.orientation[max_particle]), 0], 
                                 [sin(particles.orientation[max_particle]), cos(particles.orientation[max_particle]),  0], 
                                 [0, 0, 1]])

            stereo2world_R = rotation @ VEHICLE2STEREO['R']
            stereo2world_p = position + rotation @ VEHICLE2STEREO['T']
            world_p = stereo2world_p[:, None] + stereo2world_R @ target_T
            valid_idz = np.logical_and((world_p[-1, :] < STEREO['STEREO_Z_RANGE'][1]),
                                         (world_p[-1, :] > STEREO['STEREO_Z_RANGE'][0]))
            world_p = world_p[:-1, valid_idz]
            img = img[:, valid_idz]
            idx, idy = real2index(particles.map.map_value, particles.map.x_im, 
                                     particles.map.y_im, world_p[0, :], world_p[1, :])
            color_map[:, idx, idy] = img

            # Compute camera to world transform

    if counter % 1000 == 0:
        idx, idy = real2index(particles.map.map_value, particles.map.x_im, 
                                    particles.map.y_im, particles.position[:, 0], particles.position[:, 1])
        trace_x.append(idx)
        trace_y.append(idy)
        
    if counter % 10000 == 0:
        plt.imshow(np.sign(particles.map.map_value).T, origin='lower')
        plt.plot(np.vstack(trace_x), np.vstack(trace_y), color="red", linewidth=0.1)
        plt.axis("off")
        plt.ioff()
        plt.savefig("output/trace_{}.png".format(counter), dpi=500)
        plt.imshow(color_map.T, origin='lower')
        plt.axis("off")
        plt.ioff()
        plt.savefig("output/color_{}.png".format(counter), dpi=500)
        plt.close('all')

plt.imshow(color_map.T, origin='lower')
plt.axis("off")
plt.savefig('output/colormap.png', dpi=500)
plt.ioff()
plt.close('all')
print('------------Done------------')