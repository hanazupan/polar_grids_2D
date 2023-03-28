
import numpy as np
from numpy._typing import NDArray
from scipy.constants import pi


def from_polar_to_cartesian(rs, thetas):
    return rs * np.cos(thetas), rs * np.sin(thetas)


class PolarGrid:

    def __init__(self, r_lim: tuple = None, num_radial: int = 50, num_angular: int = 50):
        if r_lim is None:
            r_lim = (0, 10)

        self.r_lim = r_lim
        self.num_radial = num_radial
        self.num_angular = num_angular

        self.rs = np.linspace(*r_lim, num=num_radial)
        self.thetas = np.linspace(0, 2*pi, num=num_angular)

    def get_radial_grid(self, theta: float = 0):
        single_thetas = np.full(self.rs.shape, theta)
        return np.dstack((self.rs, single_thetas)).squeeze()

    def get_polar_meshgrid(self):
        return np.meshgrid(self.rs, self.thetas)

    def get_cartesian_meshgrid(self):
        mesh_rs, mesh_thetas = self.get_polar_meshgrid()
        return from_polar_to_cartesian(mesh_rs, mesh_thetas)

    def get_flattened_polar_coords(self):
        Rs, Thetas = self.get_polar_meshgrid()
        return np.vstack([Rs.ravel(), Thetas.ravel()]).T

    def get_flattened_cartesian_coords(self):
        xs, ys = self.get_cartesian_meshgrid()
        return np.vstack([xs.ravel(), ys.ravel()]).T

if __name__ == "__main__":
    pg = PolarGrid(r_lim=(0,8), num_radial=5, num_angular=7)
    print(pg.get_polar_meshgrid())
    print(pg.get_cartesian_meshgrid())

