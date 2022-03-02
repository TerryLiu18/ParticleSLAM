"""
cluster of particles
"""

from math import *
from cv2 import threshold
import numpy as np
import matplotlib.pyplot as plt 
from data import *
from pr2_utils import *
from map import Map


def one_degree_orientation(orient):
    m_cos, m_sin = np.cos(orient), np.sin(orient)
    return np.transpose(np.array([[m_cos, -m_sin], [m_sin, m_cos]]),axes=[2, 0, 1])


class Particles:
    def __init__(self, map, n_particles, n_efficient_threshold):
        """
        :param n: number of particles
        """
        self.n_particles = n_particles
        self.position = np.zeros((n_particles, 2))
        self.orientation = np.zeros((n_particles, 1))
        self.weights = np.ones(n_particles) / n_particles
        self.forward_noise = 1e-10
        self.turn_noise    = 1e-11
        self.sense_noise   = 0.0
        self.n_efficient_threshold = 0.1
        self.map = map
    
    def car_state(self):
        return np.sum(self.weights.reshape(-1, 1) * self.position, axis=0)

    """
    step1: predict: move the particles using the differential drive model
    step2: update weight: sense the environment (using mapCorrelation) and update the weights
    step3: update map: choose the largest prob particle and convert laser scan to map index and update the map
    step3: resampling: resample the particles according to the weights
    """

    def predict(self, velocity, angle_velocity, dt):
        """
        predict next state of the particle
        :param velocity: velocity of the particles
        :param orientation: orientation of the particles
        :param dt: time interval (time step: nano second)
        :return: position
        :return: orientation
        """ 
        velocity += np.random.normal(scale=self.forward_noise, size=self.orientation.shape)
        angle_velocity += np.random.normal(scale=self.turn_noise, size=self.orientation.shape)
        self.position = self.position + np.hstack([velocity*np.cos(self.orientation),
                                                    velocity*np.sin(self.orientation)]) * dt
        self.orientation = self.orientation + angle_velocity * dt 

    def update_weights(self, lidar_data):
        """
        update the weights of the particles
        Sw = R @ Sb + P

        some notations:
        l: lidar
        v: vehicle (body frame or particle frame)
        w: world

        :param lidar: a line of lidar points, (286, )
        """

        particle_weights = np.zeros_like(self.weights)
        particle_R = one_degree_orientation(self.orientation.reshape(-1))  # here particle stands for 
        particle_p = self.position  # n * 2
        world2lidar_R = particle_R @ LIDAR_PARAMETER['R'][0:2, 0:2]
        world2lidar_p = particle_p + particle_R @ LIDAR_PARAMETER['T'][0:2]

        rad = np.linspace(-5, 185, num=286) / 180 * pi
        lidar_p = np.vstack([lidar_data * np.cos(rad), lidar_data * np.sin(rad)])
        world_p = world2lidar_p[:, :, None] + world2lidar_R @ lidar_p

        for particle in range(self.n_particles):
            wo_p = world_p[particle, :, :]
            co_x = np.linspace(np.zeros_like(wo_p[0, :]), wo_p[0, :], num=3)
            co_y = np.linspace(np.zeros_like(wo_p[1, :]), wo_p[1, :], num=3)
            obstacle = np.zeros_like(co_x)
            obstacle[-1, :][np.where(lidar_data == 80)[0]] = 1
            particle_weights[particle] = mapCorrelation(self.map.map_value, self.map.x_im,  self.map.y_im, 
                                            np.vstack([co_x.reshape(-1), co_y.reshape(-1)]), obstacle.reshape(-1))
        self.weights = self.weights * np.exp(particle_weights-np.max(particle_weights))
        self.weights /= np.sum(self.weights)

    def resample(self):
        # update using resampling
        n_eff = 1 / np.sum(self.weights ** 2)
        if n_eff / self.n_particles < self.n_efficient_threshold:
            # idx = self.resample()
            idx = np.random.choice(range(self.n_particles), size=self.n_particles, replace=True, p=self.weights)
            self.position = self.position[idx, :]
            self.orientation = self.orientation[idx, :]
            self.weights = self.weights[idx]  
            self.weights /= np.sum(self.weights) 

    def update_map(self, idx, lidar_point):
        """
        update map
        :param world_index: world coordinate to map index, shape: (2, n)
        :param l2w_p: lidar to world position, shape: (1, 3)
        :return: updated map
        """

        body_R = np.array([[cos(self.orientation[idx]), -sin(self.orientation[idx]), 0], 
                                  [sin(self.orientation[idx]),  cos(self.orientation[idx]), 0], 
                                  [0, 0, 1]])
        body_p = np.array([self.position[idx, 0], self.position[idx, 1], 0])

        world2lidar_R = body_R @ LIDAR_PARAMETER['R']
        world2lidar_p = body_p + body_R @ LIDAR_PARAMETER['T']

        rad = np.linspace(-5, 185, num=286) / 180 * pi
        max_ranges = np.where(lidar_point == LIDAR_PARAMETER['max_range'])[0]

        lidar_p = np.vstack([lidar_point * np.cos(rad), lidar_point * np.sin(rad), np.zeros_like(lidar_point)])
        world_p = world2lidar_p[:, None] + world2lidar_R @ lidar_p

        # prepare for Bresenham2D
        idx, idy = real2index(self.map.map_value, self.map.x_im, self.map.y_im, world_p[0, :], world_p[1, :])
        x_0, y_0 = real2index(self.map.map_value, self.map.x_im, self.map.y_im, world2lidar_p[0], world2lidar_p[1])
        for i, (x, y) in enumerate(zip(idx, idy)):
            path_idx = bresenham2D(x_0, y_0, x, y)
            path_idx = path_idx.astype(np.int16)
            self.map.map_value[path_idx[0, :-1], path_idx[1, :-1]] -= log(4)
            if i not in max_ranges:
                self.map.map_value[path_idx[0, -1], path_idx[1, -1]] += log(4)
        np.clip(self.map.map_value, -10, 10)

    def show_particles(self, id):
        """
        print out particles`    
        """
        x, y = self.position[:, 0], self.position[:, 1]
        index = self.map.coordinate_to_index(x, y)
        x_index, y_index = index[0, :], index[1, :]
        plt.plot(x_index, y_index, '.')
        plt.savefig('./img/particles_{}.png'.format(id))


if __name__ == '__main__':
    # test Particles class
    particles = Particles(map = Map((-100, 100), (-100, 100), 1), n_particles=4, n_efficient_threshold=0.1)
    time_stamp, lidar_points = read_data_from_csv('./data/sensor_data/lidar.csv')
    particles.update_weights(np.array(lidar_points[0, :]))
    