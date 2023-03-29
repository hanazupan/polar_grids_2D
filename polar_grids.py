"""
Polar coordinates are defined with a radius and an angle.

A polar grid is a meshgrid of radial and angular discretisation (in equidistant steps).

Meshgrids are always given in the numpy convention.

Flat coordinate grids are given as arrays of coordinate pairs in the following convention:

[[r0, theta0],                           [[x0, y0],
 [r1, theta1],                            [x1, y1],
 [r2, theta2],             or             [x2, y2],
  .....                                    ......
 [rN, thetaN]]                            [xN, yN]]
"""

import numpy as np
from numpy.typing import NDArray
from scipy.constants import pi


def from_polar_to_cartesian(rs: NDArray, thetas: NDArray) -> tuple[NDArray, NDArray]:
    """
    Performs coordinate transform.

    Takes an array of radii and an array of angles - they should be of the same shape - and returns the arrays of
    x and y coordinates in the same shapes.
    """
    assert rs.shape == thetas.shape, f"Arrays of coordinates must be of same shape: {rs.shape}!={thetas.shape}"
    return rs * np.cos(thetas), rs * np.sin(thetas)


class PolarGrid:

    """
    A polar grid always encompasses an entire circle (2pi rad) and a range of radii defined by r_lim. The number of
    equally spaced points in the radial and angular discretisation can be individually controlled.
    """

    def __init__(self, r_lim: tuple[float, float] = None, num_radial: int = 50, num_angular: int = 50):
        if r_lim is None:
            r_lim = (0, 10)

        self.r_lim = r_lim
        self.num_radial = num_radial
        self.num_angular = num_angular

        # 1D discretisations in radial and angular dimensions
        self.rs = np.linspace(*r_lim, num=num_radial)
        self.thetas = np.linspace(0, 2*pi, num=num_angular, endpoint=False)  # don't want endpoint because periodic!

    def get_radial_grid(self, theta: float = 0) -> NDArray:
        """
        Get the radially spaced grid. Fill the angular component with a constant value theta (need not to be a part of
        angular grid). Result in flat coordinate grid convention.
        """
        single_thetas = np.full(self.rs.shape, theta)
        return np.dstack((self.rs, single_thetas)).squeeze()

    def get_polar_meshgrid(self) -> list[NDArray, NDArray]:
        """
        Return the r- and theta-meshgrid with all combinations of coordinates.
        The result is a list of two elements, each with shape (self.num_angular, self.num_radial)
        """
        return np.meshgrid(self.rs, self.thetas)

    def get_cartesian_meshgrid(self) -> tuple[NDArray, NDArray]:
        """
        Take the polar meshgrid and convert it to a x- and y-meshgrid.
        """
        mesh_rs, mesh_thetas = self.get_polar_meshgrid()
        return from_polar_to_cartesian(mesh_rs, mesh_thetas)

    def get_flattened_polar_coords(self) -> NDArray:
        """
        Return all combinations of r- and theta coordinates in a flat coordinate pair format.
        """
        Rs, Thetas = self.get_polar_meshgrid()
        return np.vstack([Rs.ravel(), Thetas.ravel()]).T

    def get_flattened_cartesian_coords(self) -> NDArray:
        """
        Return all combinations of x- and y coordinates in a flat coordinate pair format.
        """
        xs, ys = self.get_cartesian_meshgrid()
        return np.vstack([xs.ravel(), ys.ravel()]).T
