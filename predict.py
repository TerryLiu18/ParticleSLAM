import numpy as np
from map import Map
from math import *
from particles import Particles
from matplotlib import pyplot as plt
from pr2_utils import *
from data import *
from tqdm import tqdm

map = Map((-100, 1500), (-1500, 100), 1)
particles = Particles(map=map, n_particles=20, n_efficient_threshold=0.1)

encoder_time, encoder_data = read_encoder(ENCODER_PATH)
yaw_time, yaw_data = read_fog(FOG_PATH)

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
timestamp = sorted(car_event.keys())

particles = Particles(map=map, n_particles=100, n_efficient_threshold=0.1)
last_t, velocity, angle_velocity = None, None, None

for t_idx, t in enumerate(tqdm(timestamp)):
    events = car_event[t]
    for event in events:
        event_type, event_idx = event
        # print('this is t={}, event_type={}, event_idx={}'.format(t, event_type, event_idx))
        if event_type == "encoder":
            # Update velocity
            velocity = encoder_data[event_idx]
            if (last_t is None):
                if (velocity is not None) and (angle_velocity is not None):
                    last_t = t
            else:
                particles.predict(velocity, angle_velocity, t-last_t)
                last_t = t

        elif event_type == "yaw":
            # Update angular velocity
            angle_velocity = yaw_data[event_idx]
            if (last_t is None):
                if (velocity is not None) and (angle_velocity is not None):
                    last_t = t
            else:
                particles.predict(velocity, angle_velocity, t-last_t)
                last_t = t
        
        if event_type == "yaw" or "encoder":
            if event_idx % 2000 == 0:
                particles.show_particles('{}_{}'.format(event_type, event_idx))

    