"""
A polar grid is a meshgrid of radial and angular discretisation (in equidistant steps).

This module includes:
    - conversion Cartesian/polar coordinates
    - positioning points (r, theta) in a uniform polar grid
    - calculating areas, dividing lines and distances between points

Polar coordinates are defined with a radius and an angle. The radial direction is discretised between r_min and r_max.
The angular direction is discretised on the entire circle, between (0, 2pi)

Meshgrids are always given in the numpy convention.

Flat coordinate grids are given as arrays of coordinate pairs in the following convention:

[[r0, theta0],                           [[x0, y0],
 [r1, theta1],                            [x1, y1],
 [r2, theta2],             or             [x2, y2],
  .....                                    ......
 [rN, thetaN]]                            [xN, yN]]

To create SphericalVoronoi cells from the grids, enumerate the flat cartesian version of the grid.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from copy import copy
from typing import Optional, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.constants import pi
from scipy.sparse import coo_array
from scipy.spatial import SphericalVoronoi

from molgri.space.utils import normalise_vectors


# tested
def from_polar_to_cartesian(rs: NDArray, thetas: NDArray) -> tuple[NDArray, NDArray]:
    """
    Performs coordinate transform from polar to cartesian coordinates.

    Args:
        rs: an array of any shape where each value is a particular radius
        thetas: an array of the same shape as rs givind the angle (in radian) belonging to the corresponding radius

    Returns:
        (xs, ys): two arrays of the same shape as rs, thetas where the calculated coordinate transforms are saved

    """
    assert rs.shape == thetas.shape, f"Arrays of coordinates must be of same shape: {rs.shape}!={thetas.shape}"
    return rs * np.cos(thetas), rs * np.sin(thetas)


def from_cartesian_to_polar(xs: NDArray, ys: NDArray) -> tuple[NDArray, NDArray]:
    """
    Performs coordinate transform from polar to cartesian coordinates.

    Returns:
        (rs, thetas)

    """
    assert xs.shape == ys.shape, f"Arrays of coordinates must be of same shape: {xs.shape}!={ys.shape}"
    return np.sqrt(xs**2 + ys**2), np.arctan2(ys, xs)

class Grid(ABC):

    """
    Abstract grid description defining all properties that a SqRA modell needs to know.
    """

    @abstractmethod
    def get_N_cells(self) -> int:
        """Obtain the total number of cells in the grid."""
        pass

    @abstractmethod
    def get_flattened_cartesian_coords(self) -> NDArray:
        """
        Return all combinations of x- and y coordinates in a flat coordinate pair format.
        This is the property that gets numbered.
        """
        pass

    def get_flattened_polar_coords(self) -> NDArray:
        """
        If nothing else implemented, convert cartesian ones.
        """
        cartesian_coo = self.get_flattened_cartesian_coords()
        xs = cartesian_coo.T[0]
        ys = cartesian_coo.T[1]
        rs, thetas = from_cartesian_to_polar(xs, ys)
        return np.vstack([rs.ravel(), thetas.ravel()]).T

    def _get_property_all_pairs(self, method: Callable) -> coo_array:
        """
        Helper method for any property that is dependent on a pair of indices. Examples: obtaining all distances
        between cell centers or all areas between cells. Always symmetrical - value at (i, j) equals the one at (j, i)

        Args:
            method: must have a signature (index1: int, index2: int, print_message: boolean)

        Returns:
            a sparse matrix of shape (len_flat_position_array, len_flat_position_array)
        """
        data = []
        row_indices = []
        column_indices = []
        N_pos_array = len(self.get_flattened_cartesian_coords())
        for i in range(N_pos_array):
            for j in range(i + 1, N_pos_array):
                my_property = method(i, j, print_message=False)
                if my_property is not None:
                    # this value will be added to coordinates (i, j) and (j, i)
                    data.extend([my_property, my_property])
                    row_indices.extend([i, j])
                    column_indices.extend([j, i])
        sparse_property = coo_array((data, (row_indices, column_indices)), shape=(N_pos_array, N_pos_array),
                                    dtype=float)
        return sparse_property

    @abstractmethod
    def get_all_voronoi_surfaces_as_numpy(self) -> NDArray:
        """
        If l is the length of the flattened position array, returns a lxl (sparse) array where the i-th row and j-th
        column (as well as the j-th row and the i-th column) represent the size of the Voronoi surface between points
        i and j in position grid. If the points do not share a division area, no value will be set.
        """
        pass

    @abstractmethod
    def get_all_distances_between_centers_as_numpy(self) -> NDArray:
        """
        Get a sparse matrix where for all sets of neighbouring cells the distance between Voronoi centers is provided.
        Therefore, the value at [i][j] equals the value at [j][i]. For more information, check
        self.get_distance_between_centers.
        """
        pass

    @abstractmethod
    def get_all_voronoi_volumes(self) -> NDArray:
        """
        Get an array in the same order as flat position grid, listing the volumes of Voronoi cells.
        """
        pass


class CartesianGrid(Grid):

    """
    This is a rectangular grid in 2D.
    """

    def __init__(self, x_lim: tuple[float, float] = None, y_lim: tuple[float, float] = None, num_x: int = 50,
                 num_y: int = 50):
        if x_lim is None:
            x_lim = (-4, 4)
        if y_lim is None:
            y_lim = (-4, 4)

        self.x_lim = x_lim
        self.y_lim = y_lim
        self.num_x = num_x
        self.num_y = num_y

        # 1D discretisation
        self.xs = np.linspace(*self.x_lim, num=self.num_x)
        self.ys = np.linspace(*self.y_lim, num=self.num_y)
        self.delta_x = self.xs[1] - self.xs[0]
        self.delta_y = self.ys[1] - self.ys[0]

    def get_N_cells(self):
        return self.num_x * self.num_y

    def get_flattened_cartesian_coords(self):
        xs, ys = self.get_cartesian_meshgrid()
        return np.vstack([xs.ravel(), ys.ravel()]).T

    def get_cartesian_meshgrid(self) -> list[NDArray, NDArray]:
        """
        Return the r- and theta-meshgrid with all combinations of coordinates.
        The result is a list of two elements, each with shape (self.num_angular, self.num_radial)
        """
        return np.meshgrid(self.xs, self.ys)

    def _are_sideways_neig(self, i, j):
        delta_index_one = np.abs(i-j) == 1
        not_next_row = i // self.num_x == j // self.num_x
        return delta_index_one and not_next_row

    def _are_above_below_neig(self, i, j):
        same_remainder = i % self.num_x == j % self.num_x
        delta_index_x = np.abs(i-j) == self.num_x
        return delta_index_x and same_remainder

    def get_distance_between_centers(self, index_1: int, index_2: int, print_message = True) -> Optional[float]:
        if self._are_sideways_neig(index_1, index_2):
            return self.delta_x
        if self._are_above_below_neig(index_1, index_2):
            return self.delta_y

    def get_division_area(self, index_1: int, index_2: int, print_message = True) -> Optional[float]:
        if self._are_sideways_neig(index_1, index_2):
            return self.delta_y
        if self._are_above_below_neig(index_1, index_2):
            return self.delta_x

    def get_all_voronoi_surfaces_as_numpy(self) -> NDArray:
        return self._get_property_all_pairs(self.get_division_area).toarray()


    def get_all_distances_between_centers_as_numpy(self) -> NDArray:
        return self._get_property_all_pairs(self.get_distance_between_centers).toarray()

    def get_all_voronoi_volumes(self) -> NDArray:
        """
        For a rectangular grid, all "volumes" (areas) of cells are the same
        """

        return np.full((self.get_N_cells(),), self.delta_x*self.delta_y)


class PolarGrid (Grid):

    """
    A polar grid always encompasses an entire circle (2pi rad) and a range of radii defined by r_lim. The number of
    equally spaced points in the radial and angular discretisation can be individually controlled.
    """

    def __init__(self, r_lim: tuple[float, float] = None, num_radial: int = 50, num_angular: int = 50):
        """
        Create a polar grid.

        Args:
            r_lim: (minimal radius, maximal radius) at which points are positioned
            num_radial: number of points in radial direction
            num_angular: number of points in angular direction
        """
        if r_lim is None:
            r_lim = (0.1, 10)

        self._verify_input(r_lim=r_lim, num_radial=num_radial, num_angular=num_angular)

        self.r_lim = r_lim
        self.num_radial = num_radial
        self.num_angular = num_angular
        self.N_cells = self.num_radial * self.num_angular

        # 1D discretisations in radial and angular dimensions
        self.rs = None
        self.thetas = None
        self.get_rs()
        self.get_thetas()

        # voronoi
        self.all_sv = None

    def _verify_input(self, r_lim: tuple[float, float], num_radial: int, num_angular: int):
        assert num_angular >= 1
        assert num_radial >= 1
        if r_lim[0] > r_lim[1]:
            raise ValueError("The lower limit of the radial grid must be smaller than the higher limit.")
        if num_radial == 1 and not np.allclose(r_lim[0], r_lim[1]):
            raise ValueError(f"Cannot cover the entire span of radii {r_lim} with only one radial point. Change the"
                             f"limits to a single value or increase the number of points.")

    def __str__(self):
        return f"polar_grid_{self.r_lim[0]}_{self.r_lim[1]}_{self.num_radial}_{self.num_angular}"

    def get_N_cells(self):
        return self.num_radial * self.num_angular

    #################################################################################################################
    #                                GETTERS FOR ALL SORTS OF GRID REPRESENTATIONS
    #################################################################################################################

    # tested
    def get_rs(self) -> NDArray:
        """
        Obtain a sorted array (smallest to largest) of all radii at which points are positioned.

        Returns:
            an array of point radii of length self.num_radial
        """
        if self.rs is None:
            self.rs = np.linspace(*self.r_lim, num=self.num_radial)
        assert len(self.rs) == self.num_radial
        assert np.isclose(self.rs[0], self.r_lim[0])
        assert np.isclose(self.rs[-1], self.r_lim[1])
        return self.rs

    # tested
    def get_thetas(self) -> NDArray:
        """
        Obtain a sorted array (smallest to largest) of all angles at which points are positioned (all between 0, 2pi).

        Returns:
            an array of point angles of length self.num_angular
        """
        if self.thetas is None:
            # don't want endpoint because periodic!
            self.thetas = np.linspace(0, 2*pi, num=self.num_angular, endpoint=False)
        assert len(self.thetas) == self.num_angular
        assert np.isclose(self.thetas[0], 0)
        return self.thetas

    # tested
    def get_radial_grid(self, theta: float = 0) -> NDArray:
        """
        Get an array of polar coordinates [(r1, theta), (r2, theta) ... (rN, theta)] featuring all radial distances but
        a single constant value of theta.

        Args:
            theta: a value of the angle (in rad)

        Returns:
            [(r1, theta), (r2, theta) ... (rN, theta)] an array of shape (self.num_radial, 2)

        Get the radially spaced grid. Fill the angular component with a constant value theta (need not to be a part of
        angular grid). Result in flat coordinate grid convention.
        """
        single_thetas = np.full(self.num_radial, theta)
        result = np.dstack((self.get_rs(), single_thetas)).squeeze()
        assert result.shape == (self.num_radial, 2)
        return result

    # tested
    def get_unit_angular_grid(self) -> NDArray:
        """
        Get an array of polar coordinates [(1, theta1), (1, theta2) ... (1, thetaM)] featuring all angular positions
        but all at unit distance.

        Returns:
            [(1, theta1), (1, theta2) ... (1, thetaM)] an array of shape (self.num_angular, 2)
        """
        Rs, Thetas = np.meshgrid(np.array([1.]), self.get_thetas())
        result = np.vstack([Rs.ravel(), Thetas.ravel()]).T
        assert result.shape == (self.num_angular, 2)
        return result

    # tested
    def get_unit_cartesian_grid(self) -> NDArray:
        """
        Use the self.get_unit_angular_grid as a starting point and convert it to cartesian coordinates.

        Returns:
            [[x1, y1], [x2, y2] ... [xM, yM]],
             where M is the number of angular points
        """
        mesh_rs, mesh_thetas = np.meshgrid(np.array([1.]), self.get_thetas())
        xs, ys = from_polar_to_cartesian(mesh_rs, mesh_thetas)
        result = np.vstack([xs.ravel(), ys.ravel()]).T
        assert result.shape == (self.num_angular, 2)
        return result

    def get_polar_meshgrid(self) -> list[NDArray, NDArray]:
        """
        Return the r- and theta-meshgrid with all combinations of coordinates.
        The result is a list of two elements, each with shape (self.num_angular, self.num_radial)
        """
        return np.meshgrid(self.get_rs(), self.get_thetas())

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

    # TODO: delete or improve
    def assign_to_states(self, states_list: list):
        flat_coord = self.get_flattened_polar_coords()
        all_asignments = np.zeros((len(states_list), len(flat_coord)), dtype=bool)
        for i, state in enumerate(states_list):
            all_asignments[i] = state(*flat_coord.T)
        return all_asignments

    #################################################################################################################
    #                                        VORONOI CELLS AND GEOMETRY
    #################################################################################################################

    def _index_valid(self, index: int) -> bool:
        return 0 <= index < self.N_cells

    def _index_to_layer(self, index: int) -> int:
        """
        Convert an index of flattened coordinates to the layer index they belong to.

        Args:
            index: an index of flat coordinates that identifies a cell

        Returns:
            a number between 0 and self.num_radial that indicates the layer (distance from origin) of this point

        For example in a grid of 3 radial times 5 angular points, the layers are:
        [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        """
        if not self._index_valid(index):
            return np.NaN
        return index % self.num_radial

    def _index_to_slice(self, index: int) -> int:
        """
        Convert an index of flattened coordinates to the slice index they belong to.

        Args:
            index: an index of flat coordinates that identifies a cell

        Returns:
            a number between 0 and self.num_angular that indicates the slice (ray index) of this point

        For example in a grid of 3 radial times 5 angular points, the slices are:
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        """
        if not self._index_valid(index):
            return np.NaN
        return index // self.num_radial

    def get_all_indices_of_layer_i(self, i: int) -> list:
        """
        Get a list of flat grid indices that all belong to i-th layer

        Args:
            i: layer index between 0 and self.num_radial (end non-inclusive)

        Returns:
            a list of indices, all are values between 0 and self.N_cells (end non-inclusive)
        """
        all_indices = list(range(0, self.N_cells))
        selected_indices = []
        for my_ind in all_indices:
            if self._index_to_layer(my_ind) == i:
                selected_indices.append(my_ind)
        return selected_indices

    def get_all_indices_of_slice_j(self, j):
        """
        Get a list of flat grid indices that all belong to j-th row

        Args:
            j: ray index between 0 and self.num_angular (end non-inclusive)

        Returns:
            a list of indices, all are values between 0 and self.N_cells (end non-inclusive)
        """
        all_indices = list(range(0, self.N_cells))
        selected_indices = []
        for my_ind in all_indices:
            if self._index_to_slice(my_ind) == j:
                selected_indices.append(my_ind)
        return selected_indices

    def _are_sideways_neighbours(self, index_1: int, index_2: int):
        if not self._index_valid(index_1) or not self._index_valid(index_2):
            raise AttributeError("Index beyond the scope of the grid.")
        same_layer = self._index_to_layer(index_1) == self._index_to_layer(index_2)
        slice_1 = self._index_to_slice(index_1)
        slice_2 = self._index_to_slice(index_2)
        neighbour_slice = np.abs(slice_2 - slice_1) == 1 or np.abs(slice_2 - slice_1) == self.num_angular - 1
        return same_layer and neighbour_slice

    def _are_ray_neighbours(self, index_1: int, index_2: int):
        if not self._index_valid(index_1) or not self._index_valid(index_2):
            raise AttributeError("Index beyond the scope of the grid.")
        same_slice = self._index_to_slice(index_1) == self._index_to_slice(index_2)
        neighbour_layer = np.abs(self._index_to_layer(index_1) - self._index_to_layer(index_2)) == 1
        return same_slice and neighbour_layer

    def get_between_radii(self):
        radii = self.get_radial_grid().T[0]
        # get increments to each radius, remove first one and add an extra one at the end with same distance as
        # second-to-last one
        increments = [radii[i]-radii[i-1] for i in range(1, len(radii))]
        if len(increments) > 1:
            increments.append(increments[-1])
            increments = np.array(increments)
            increments = increments / 2
        else:
            increments = np.array(increments)

        between_radii = radii + increments
        return between_radii

    def get_distance_between_centers(self, index_1: int, index_2: int, print_message=True) -> Optional[float]:
        """From two indices (indicating position in flat grid), determine the straight or curved distance between their
        centers."""
        if self._are_sideways_neighbours(index_1, index_2):
            layer = self._index_to_layer(index_1)
            radius = self.get_rs()[layer]
            return 2*pi*radius/self.num_angular
        elif self._are_ray_neighbours(index_1, index_2):
            r_1 = self.get_rs()[self._index_to_layer(index_1)]
            r_2 = self.get_rs()[self._index_to_layer(index_2)]
            return np.abs(r_2 - r_1)
        elif print_message:
            print(f"Points {index_1} and {index_2} are not neighbours.")

    def get_division_area(self, index_1: int, index_2: int, print_message: bool = True) -> Optional[float]:
        """From two indices (indicating position in flat grid), determine the straight or curved area between both
        cells."""
        if self._are_ray_neighbours(index_1, index_2):
            layer_1 = self._index_to_layer(index_1)
            layer_2 = self._index_to_layer(index_2)
            radius = self.get_between_radii()[np.min([layer_1, layer_2])]
            return 2 * pi * radius / self.num_angular
        elif self._are_sideways_neighbours(index_1, index_2):
            r_above = self.get_between_radii()[self._index_to_layer(index_1)]
            if self._index_to_layer(index_1) == 0:
                return r_above
            else:
                return r_above - self.get_between_radii()[self._index_to_layer(index_1)-1]
        elif print_message:
            print(f"Points {index_1} and {index_2} are not neighbours.")

    def _change_voronoi_radius(self, sv: SphericalVoronoi, new_radius: float) -> SphericalVoronoi:
        """
        This is a helper function. Since a FullGrid consists of several layers of spheres in which the points are at
        exactly same places (just at different radii), it makes sense not to recalculate, but just to scale the radius,
        vertices and points out of which the SphericalVoronoi consists to a new radius.
        """
        sv.radius = new_radius
        sv.vertices = normalise_vectors(sv.vertices, length=new_radius)
        sv.points = normalise_vectors(sv.points, length=new_radius)
        # important that it's a copy!
        return copy(sv)

    def get_voronoi_discretisation(self) -> list[SphericalVoronoi]:
        """
        Create a list of spherical voronoi-s that are identical except at different radii. The radii are set in such a
        way that there is always a Voronoi cell layer right in-between two layers of grid cells. (see FullGrid method
        get_between_radii for details.
        """
        if self.all_sv is None:
            unit_sph_voronoi = SphericalVoronoi(self.get_unit_cartesian_grid())
            between_radii = self.get_between_radii()
            self.all_sv = [self._change_voronoi_radius(unit_sph_voronoi, r) for r in between_radii]
        return self.all_sv

    def get_volume(self, index: int) -> float:
        """
        Get the volume of any cell in FullVoronoiGrid, defined by its index in flattened position grid.
        """
        return self.get_all_voronoi_volumes()[index]

    def get_all_voronoi_volumes(self) -> NDArray:
        """
        Get an array in the same order as flat position grid, listing the volumes of Voronoi cells.
        """
        vor_radius = list(self.get_between_radii())
        vor_radius.insert(0, 0)
        vor_radius = np.array(vor_radius)
        ideal_volumes = pi * (vor_radius[1:] ** 2 - vor_radius[:-1] ** 2) / self.num_angular
        volumes = []
        for i, point in enumerate(self.get_flattened_cartesian_coords()):
            layer = i % self.num_radial
            volumes.append(ideal_volumes[layer])

        volumes = np.array(volumes)
        return volumes

    def _test_pair_property(self, i, j, print_message=False):
        return i+j

    def get_all_voronoi_surfaces(self) -> coo_array:
        """
        If l is the length of the flattened position array, returns a lxl (sparse) array where the i-th row and j-th
        column (as well as the j-th row and the i-th column) represent the size of the Voronoi surface between points
        i and j in position grid. If the points do not share a division area, no value will be set.
        """
        return self._get_property_all_pairs(self.get_division_area)

    def get_all_distances_between_centers(self) -> coo_array:
        """
        Get a sparse matrix where for all sets of neighbouring cells the distance between Voronoi centers is provided.
        Therefore, the value at [i][j] equals the value at [j][i]. For more information, check
        self.get_distance_between_centers.
        """
        return self._get_property_all_pairs(self.get_distance_between_centers)

    def get_all_voronoi_surfaces_as_numpy(self) -> NDArray:
        """See self.get_all_voronoi_surfaces, only transforms sparse array to normal array."""
        sparse = self.get_all_voronoi_surfaces()
        dense = sparse.toarray()
        dense[np.isclose(dense, 0)] = np.nan
        return dense

    def get_all_distances_between_centers_as_numpy(self) -> NDArray:
        """See self.get_all_distances_between_centers, only transforms sparse array to normal array."""
        sparse = self.get_all_distances_between_centers()
        dense = sparse.toarray()
        dense[np.isclose(dense, 0)] = np.nan
        return dense