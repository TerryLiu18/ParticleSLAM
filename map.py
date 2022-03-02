import numpy as np
import matplotlib.pyplot as plt
from data import *
from pr2_utils import *

class Map:
    def __init__(self, xlim, ylim, acc):
        self.xmin, self.xmax = xlim
        self.ymin, self.ymax = ylim
        self.acc = acc
        self.x_im = np.linspace(self.xmin, self.xmax, int((self.xmax - self.xmin)/acc + 1))
        self.y_im = np.linspace(self.ymin, self.ymax, int((self.ymax - self.ymin)/acc + 1))
        self.nx = len(self.x_im)
        self.ny = len(self.y_im)
        self.map_value = np.zeros((self.nx, self.ny))

    def occupancy_grid(self, threshold=0.5):
        return (self.map_value > threshold).astype(np.int32)

    def probmap(self):
        prob = np.exp(self.map_value)
        return prob / (1 + prob)

    def coordinate_to_index(self, x, y):
        """
        transfer from sensor data to map index
        :param xlim: (xmin, xmin+acc, xmin+2acc... xmax)
        :param ylim: (ymin, ymin+acc, ymin+2acc... ymax)
        :param x: x coordinate (in real world), shape: (n,)
        :param y: y coordinate (in real world), shape: (n,)
        :return: (2, n)
        """

        xmin, xmax = self.x_im[0], self.x_im[-1]
        ymin, ymax = self.y_im[0], self.y_im[-1]
        nx, ny = len(self.x_im), len(self.y_im)
        res_x, res_y = (xmax-xmin)/(nx-1), (ymax-ymin)/(ny-1)
        index =  np.vstack([
            np.ceil((x - self.xmin) / res_x).reshape(1, -1),
            np.ceil((y - self.ymin) / res_y).reshape(1, -1),
        ]).astype(np.int16)
        np.clip(index[0, :], 0, nx-1)
        np.clip(index[1, :], 0, ny-1)
        return index

    def draw_map(self, id):
        plt.imshow(self.probmap())
        plt.savefig('map_img/map_{}.png'.format(id))
        # plt.show()


    def __repr__(self) -> str:
        plt.imshow(self.probmap(self.map))
        plt.show()
        return 'Map(xmin={}, xmax={}, ymin={}, ymax={}, acc={})'.format(
            self.xmin, self.xmax, self.ymin, self.ymax, self.acc
        )
    
