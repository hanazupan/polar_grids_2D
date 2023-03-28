"""
This module implements different two-dimensional potentials with a radial and circular component. Most are variations
of a double well potential where every minimum forms a circle around the origin.
"""
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.constants import pi


class AnalyticalCircularPotential(ABC):

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def get_potential(self, circ_coordinates: NDArray[float]):
        """
        Get potential at circular coordinates (r, theta) or an array of coordinate pairs that are of equal length

        Args:
            circ_coordinates: an array of coordinates in circular coordinates like

                                [[r1, theta1],
                                 [r2, theta2],
                                 [r3, theta3],
                                  ...
                                 [rN, thetaN]]

                              where r is the distance from origin and theta the angle in radians. Angle 0 is defined at
                              the line of the vector (1, 0), straight to the right in x direction
        """
        coord_shape = circ_coordinates.shape
        assert len(coord_shape) == 2 and coord_shape[1] == 2, f"Must have shape (N, 2), not {coord_shape}"

    def get_potential_as_meshgrid(self, r_meshgrid: NDArray, theta_meshgrid: NDArray):
        r_flat = np.reshape(r_meshgrid, (-1,))
        theta_flat = np.reshape(theta_meshgrid, (-1,))
        flat_coord = np.dstack((r_flat, theta_flat)).squeeze()
        pot = self.get_potential(flat_coord)
        return pot.reshape(r_meshgrid.shape)


class FlatSymmetricalDoubleWell(AnalyticalCircularPotential):

    """
    This is a very simple extension of the symmetrical (equal depth) double well potential known from 1D.
    Looking at the system radially, there are two minima and the barrier between them. There is no circular
    component, the entire circle has the same
    """

    def __init__(self, steepness: float, first_min_r: float, second_min_r: float, **kwargs):
        super().__init__(**kwargs)
        self.steepness = steepness
        assert self.steepness > 0
        self.first_min_r = first_min_r
        assert self.first_min_r >= 0
        self.second_min_r = second_min_r
        assert self.second_min_r >= 0

    def get_potential(self, circ_coordinates: NDArray[float]):
        """
        Here implement a very simple equal depth double well. The angle theta plays no role.

        Equation: A(x-m1)(x-m2) + C
        where A > 0 is the steepness of potential walls, m1 and m2 control both minima and C the min value in
        those minima.
        """
        super().get_potential(circ_coordinates)

        rs = circ_coordinates.T[0]
        return self.steepness * ((rs - self.first_min_r)**2 - self.second_min_r)**2