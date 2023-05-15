"""
This module implements different two-dimensional potentials with a radial and circular component. Most are variations
of a double well potential where every minimum forms a circle around the origin.
"""
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy.constants import R, pi


class AnalyticalCircularPotential(ABC):

    """
    This class of potentials is defined on the entire 2D space using polar coordinates r and theta. Each sub-class
    has a set of parameters that define this type of analytical potential (eg location, number and depth of minima).
    """

    def get_name(self):
        return "analytical_potential"

    def _check_coordinate_dim(self, coords: ArrayLike):
        # several coordinate pairs provided at once
        if len(coords) > 2:
            coord_shape = coords.shape
            assert len(coord_shape) == 2 and coord_shape[1] == 2, f"Must have shape (N, 2), not {coord_shape}"
        # a pair of coordinates provided - should be transformed to same format
        else:
            coords = np.array(coords)
            coords.reshape((1, 2))
        return coords

    @abstractmethod
    def get_potential_polar_coord(self, circ_coordinates: ArrayLike):
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

        Unit of potential is kJ/mol.
        """
        return self._check_coordinate_dim(circ_coordinates)

    def _meshgrid2flat(self, flat_function: Callable, r_meshgrid: NDArray, theta_meshgrid: NDArray):
        r_flat = np.reshape(r_meshgrid, (-1,))
        theta_flat = np.reshape(theta_meshgrid, (-1,))
        flat_coord = np.dstack((r_flat, theta_flat)).squeeze()
        pot = flat_function(flat_coord)
        return pot.reshape(r_meshgrid.shape)

    def get_potential_as_meshgrid(self, r_meshgrid: NDArray, theta_meshgrid: NDArray) -> NDArray:
        """
        Use a meshgrid of coordinates and get a grid of potentials in the same shape back. Useful for 3D plots.
        """
        return self._meshgrid2flat(self.get_potential_polar_coord, r_meshgrid, theta_meshgrid)

    def get_population_as_meshgrid(self, r_meshgrid: NDArray, theta_meshgrid: NDArray) -> NDArray:
        return self._meshgrid2flat(self.get_population, r_meshgrid, theta_meshgrid)

    def get_population(self, circ_coordinates: ArrayLike, T: float = 300):
        """Using the Boltzmann factor, calculate populations instead of potentials:

        pi_i = 1/Z * e^(-V_i/(RT)
        """
        not_norm_pop = np.exp(-self.get_potential_polar_coord(circ_coordinates) * 1000 / (R * T))
        pop_sum = np.sum(not_norm_pop)
        return 1/pop_sum*not_norm_pop


class FlatSymmetricalDoubleWell(AnalyticalCircularPotential):

    """
    This is a very simple extension of the symmetrical (equal depth) double well potential known from 1D.
    Looking at the system radially, there are two minima and the barrier between them. There is no circular
    component, the entire circle has the same potential.
    """

    def __init__(self, steepness: float = 10, first_min_r: float = 2, second_min_r: float = 1, **kwargs):
        super().__init__(**kwargs)
        self.steepness = steepness
        assert self.steepness > 0
        self.first_min_r = first_min_r
        assert self.first_min_r >= 0
        self.second_min_r = second_min_r
        assert self.second_min_r >= 0

    def get_name(self):
        return f"flat_sym_dw_{self.steepness}_{self.first_min_r}_{self.second_min_r}"

    def get_potential_polar_coord(self, circ_coordinates: ArrayLike):
        """
        Here implement a very simple equal depth double well. The angle theta plays no role.

        Equation: A((x-m1)^2 - m2)^2

        where A > 0 is the steepness of potential walls, m1 and m2 control both minima.
        The unit of potential is kJ/mol
        """
        circ_coordinates = super().get_potential_polar_coord(circ_coordinates)
        rs = circ_coordinates.T[0]
        return self.steepness * ((rs - self.first_min_r)**2 - self.second_min_r)**2 + 3*rs


class FlatDoubleWellAlpha(FlatSymmetricalDoubleWell):

    def __init__(self, alpha: float, exp_factor: float = 20, exp_min: float = 2, steepness: float = 10,
                 first_min_r: float = 2, second_min_r: float = 1,
                 **kwargs):
        super().__init__(steepness, first_min_r, second_min_r, **kwargs)
        self.alpha = alpha
        self.exp_factor = exp_factor
        self.exp_min = exp_min

    def get_name(self):
        return f"flat_dw_alpha_{self.alpha}"

    def get_potential_polar_coord(self, circ_coordinates: ArrayLike):
        dw_part = super().get_potential_polar_coord(circ_coordinates)
        rs = circ_coordinates.T[0]
        exp_part = self.alpha * np.exp(-self.exp_factor*(rs-self.exp_min)**2)
        return dw_part + exp_part

class RadialAndAngularWell(FlatSymmetricalDoubleWell):

    def __init__(self, k=5, f =3, x0 = 0, x1 = 2, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.f = f
        self.x0 = x0
        self.x1 = x1

    def get_potential_polar_coord(self, circ_coordinates: ArrayLike):
        thetas = circ_coordinates.T[1]
        radial_part = super(RadialAndAngularWell, self).get_potential_polar_coord(circ_coordinates)
        angular_part = self.k / 2 * (1 + np.cos(thetas - self.x0) + self.f*np.cos(thetas - self.x1))
        return radial_part + angular_part


class RadialMinDoubleWellAlpha(FlatDoubleWellAlpha):
    #TODO: needs to be periodic!

    def __init__(self, alpha: float, radial_steepness: float = 0.5, radial_min1: float = 0.7,
                 radial_min2: float = 4.3, exp_factor: float = 20, exp_min: float = 2, steepness: float = 10,
                 first_min_r: float = 2, second_min_r: float = 1, **kwargs):
        super().__init__(alpha, exp_factor, exp_min, steepness, first_min_r, second_min_r, **kwargs)
        self.radial_steepness = radial_steepness
        self.radial_min1 = radial_min1
        self.radial_min2 = radial_min2

    def get_potential_polar_coord(self, circ_coordinates: ArrayLike):
        radial_part = super().get_potential_polar_coord(circ_coordinates)
        thetas = circ_coordinates.T[1]
        theta_part = self.radial_steepness*((thetas-self.radial_min1)**2 - self.radial_min2)**2
        return radial_part * theta_part