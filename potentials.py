"""
This module implements different two-dimensional potentials with a radial and circular component. Most are variations
of a double well potential where every minimum forms a circle around the origin.
"""
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray, ArrayLike


class AnalyticalCircularPotential(ABC):

    """
    This class of potentials is defined on the entire 2D space using polar coordinates r and theta. Each sub-class
    has a set of parameters that define this type of analytical potential (eg location, number and depth of minima).
    """

    def get_name(self):
        return "analytical_potential"

    @abstractmethod
    def get_potential(self, circ_coordinates: ArrayLike):
        """
        Get potential at circular coordinates (r, theta) or an array of coordinate pairs

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
        # several coordinate pairs provided at once
        if len(circ_coordinates) > 2:
            coord_shape = circ_coordinates.shape
            assert len(coord_shape) == 2 and coord_shape[1] == 2, f"Must have shape (N, 2), not {coord_shape}"
        # a pair of coordinates provided - should be transformed to same format
        else:
            circ_coordinates = np.array(circ_coordinates)
            circ_coordinates.reshape((1, 2))
        return circ_coordinates

    def get_potential_as_meshgrid(self, r_meshgrid: NDArray, theta_meshgrid: NDArray) -> NDArray:
        """
        Use a meshgrid of coordinates and get a grid of potentials in the same shape back. Useful for 3D plots.
        """
        r_flat = np.reshape(r_meshgrid, (-1,))
        theta_flat = np.reshape(theta_meshgrid, (-1,))
        flat_coord = np.dstack((r_flat, theta_flat)).squeeze()
        pot = self.get_potential(flat_coord)
        return pot.reshape(r_meshgrid.shape)


class FlatSymmetricalDoubleWell(AnalyticalCircularPotential):

    """
    This is a very simple extension of the symmetrical (equal depth) double well potential known from 1D.
    Looking at the system radially, there are two minima and the barrier between them. There is no circular
    component, the entire circle has the same potential.
    """

    def __init__(self, steepness: float, first_min_r: float, second_min_r: float, **kwargs):
        super().__init__(**kwargs)
        self.steepness = steepness
        assert self.steepness > 0
        self.first_min_r = first_min_r
        assert self.first_min_r >= 0
        self.second_min_r = second_min_r
        assert self.second_min_r >= 0

    def get_name(self):
        return f"flat_sym_dw_{self.steepness}_{self.first_min_r}_{self.second_min_r}"

    def get_potential(self, circ_coordinates: ArrayLike):
        """
        Here implement a very simple equal depth double well. The angle theta plays no role.

        Equation: A((x-m1)^2 - m2)^2

        where A > 0 is the steepness of potential walls, m1 and m2 control both minima.
        """
        circ_coordinates = super().get_potential(circ_coordinates)
        rs = circ_coordinates.T[0]
        return self.steepness * ((rs - self.first_min_r)**2 - self.second_min_r)**2
