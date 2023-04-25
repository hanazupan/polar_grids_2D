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

To create SphericalVoronoi cells from the grids, enumerate the flat cartesian version of the grid.
"""
from __future__ import annotations
from copy import copy
from typing import Optional, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.constants import pi
from scipy.sparse import coo_array
from scipy.spatial import SphericalVoronoi


from molgri.space.utils import  normalise_vectors


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
            r_lim = (0.1, 10)

        self.r_lim = r_lim
        self.num_radial = num_radial
        self.num_angular = num_angular
        self.N_cells = self.num_radial * self.num_angular

        # 1D discretisations in radial and angular dimensions
        self.rs = np.linspace(*r_lim, num=num_radial)
        self.thetas = np.linspace(0, 2*pi, num=num_angular, endpoint=False)  # don't want endpoint because periodic!

    def get_name(self):
        return f"polar_grid_{self.r_lim[0]}_{self.r_lim[1]}_{self.num_radial}_{self.num_angular}"

    def get_radial_grid(self, theta: float = 0) -> NDArray:
        """
        Get the radially spaced grid. Fill the angular component with a constant value theta (need not to be a part of
        angular grid). Result in flat coordinate grid convention.
        """
        single_thetas = np.full(self.rs.shape, theta)
        return np.dstack((self.rs, single_thetas)).squeeze()

    def get_unit_angular_grid(self):
        Rs, Thetas = np.meshgrid(np.array([1.]), self.thetas)
        return np.vstack([Rs.ravel(), Thetas.ravel()]).T

    def get_unit_cartesian_grid(self):
        mesh_rs, mesh_thetas = np.meshgrid(np.array([1.]), self.thetas)
        xs, ys = from_polar_to_cartesian(mesh_rs, mesh_thetas)
        return np.vstack([xs.ravel(), ys.ravel()]).T

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

    #################################################################################################################
    #                                        VORONOI CELLS AND GEOMETRY
    #################################################################################################################

    def _index_valid(self, index):
        return index < self.num_angular * self.num_radial

    def _index_to_layer(self, index):
        if not self._index_valid(index):
            return np.NaN
        return index % self.num_radial

    def _index_to_slice(self, index):
        if not self._index_valid(index):
            return np.NaN
        return index // self.num_radial

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

    def get_full_voronoi_grid(self):
        return FlatVoronoiGrid(self)

    def get_distance_between_centers(self, index_1: int, index_2: int, print_message=True) -> Optional[float]:
        """From two indices (indicating position in flat grid), determine the straight or curved distance between their
        centers."""
        if self._are_sideways_neighbours(index_1, index_2):
            layer = self._index_to_layer(index_1)
            radius = self.rs[layer]
            return 2*pi*radius/self.num_angular
        elif self._are_ray_neighbours(index_1, index_2):
            r_1 = self.rs[self._index_to_layer(index_1)]
            r_2 = self.rs[self._index_to_layer(index_2)]
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


class FlatVoronoiGrid:

    """
   Equal to FullVoronoiGrid, but in 2D. Voranoi cells are now circular archs, again positioned at mid-radii

    Enables calculations of all distances, areas, and volumes of/between cells.
    """

    def __init__(self, polar_grid: PolarGrid):
        self.full_grid = polar_grid
        self.flat_positions = self.full_grid.get_flattened_cartesian_coords()
        self.all_sv = None
        self.get_voronoi_discretisation()

    ###################################################################################################################
    #                           basic creation/getter functions
    ###################################################################################################################

    def get_name(self) -> str:
        """Name for saving files."""
        return f"voronoi_{self.full_grid.get_name()}"

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
            unit_sph_voronoi = SphericalVoronoi(self.full_grid.get_unit_cartesian_grid())
            between_radii = self.full_grid.get_between_radii()
            self.all_sv = [self._change_voronoi_radius(unit_sph_voronoi, r) for r in between_radii]
        return self.all_sv

    ###################################################################################################################
    #                           point helper functions
    ###################################################################################################################

    # def _at_same_radius(self, index1: int, index2: int) -> bool:
    #     """Check that two points belong to the same layer"""
    #     point1 = self.full_grid.get_flattened_cartesian_coords()[index1]
    #     point2 = self.full_grid.get_flattened_cartesian_coords()[index2]
    #     return bool(np.isclose(np.linalg.norm(point1), np.linalg.norm(point2)))
    #
    # def _are_sideways_neighbours(self, index1: int, index2: int) -> bool:
    #     """
    #     Check whether points belong to the same radius and are neighbours.
    #     """
    #     next_to = np.abs(index1-index2) == self.full_grid.num_radial or \
    #               np.abs(index1-index2) == self.full_grid.num_radial * (self.full_grid.num_angular - 1)
    #     return self._at_same_radius(index1, index2) and next_to
    #
    # def _are_on_same_ray(self, point1: Point2D, point2: Point2D) -> bool:
    #     """
    #     Two points are on the same ray if they are on the same vector from origin (may be at different distances)
    #     """
    #     normalised1 = point1.get_normalised_point()
    #     normalised2 = point2.get_normalised_point()
    #     return np.allclose(normalised1, normalised2)

    # def _point1_right_above_point2(self, index1: int, index2: int) -> bool:
    #     """
    #     Right above should be interpreted spherically.
    #
    #     Conditions: on the same ray + radius of point1 one unit bigger
    #     """
    #     radial_index1 = self.full_grid._index_to_layer(index1)
    #     radial_index2 = self.full_grid._index_to_layer(index2)
    #     return self._are_on_same_ray(point1, point2) and radial_index1 == radial_index2 + 1
    #
    # def _point2_right_above_point1(self, point1: Point2D, point2: Point2D) -> bool:
    #     """See _point1_right_above_point2."""
    #     radial_index1 = point1.index_radial
    #     radial_index2 = point2.index_radial
    #     return self._are_on_same_ray(point1, point2) and radial_index1 + 1 == radial_index2

    ###################################################################################################################
    #                           useful properties
    ###################################################################################################################

    # def find_voronoi_vertices_of_point(self, point_index: int, which: str = "all") -> NDArray:
    #     """
    #     Using an index (from flattened position grid), find which voronoi vertices belong to this point.
    #
    #     Args:
    #         point_index: for which point in flattened position grid the vertices should be found
    #         which: which vertices: all, upper or lower
    #
    #     Returns:
    #         an array of vertices, each row is a 3D point.
    #     """
    #     my_point = Point2D(point_index, self)
    #
    #     if which == "all":
    #         vertices = my_point.get_vertices()
    #     elif which == "upper":
    #         vertices = my_point.get_vertices_above()
    #     elif which == "lower":
    #         vertices = my_point.get_vertices_below()
    #     else:
    #         raise ValueError("The value of which not recognised, select 'all', 'upper', 'lower'.")
    #     return vertices

    # def get_distance_between_centers(self, index_1: int, index_2: int, print_message=True) -> Optional[float]:
    #     """
    #     Calculate the distance between two position grid points selected by their indices. Optionally print message
    #     if they are not neighbours.
    #
    #     There are three options:
    #         - point1 is right above point2 or vide versa -> the distance is measured in a straight line from the center
    #         - point1 and point2 are sideways neighbours -> the distance is measured on the circumference of their radius
    #         - point1 and point2 are not neighbours -> return None
    #
    #     Returns:
    #         None if not neighbours, distance in angstrom else
    #     """
    #     circular_distances = 2 * pi * self.full_grid.rs / self.full_grid.num_angular
    #
    #     point_1 = Point2D(index_1, self)
    #     point_2 = Point2D(index_2, self)
    #
    #     if self._point1_right_above_point2(point_1, point_2) or self._point2_right_above_point1(point_1, point_2):
    #         return np.abs(point_1.d_to_origin - point_2.d_to_origin)
    #     elif self._are_sideways_neighbours(index_1, index_2):
    #         radial_index = point_1.index_radial
    #         return circular_distances[radial_index]
    #     else:
    #         if print_message:
    #             print(f"Points {index_1} and {index_2} are not neighbours.")
    #         return None
    #
    # def get_division_area(self, index_1: int, index_2: int, print_message: bool = True) -> Optional[float]:
    #     """
    #     Calculate the area (in Angstrom squared) that is the border area between two Voronoi cells. This is either
    #     a curved area (a part of a sphere) if the two cells are one above the other or a flat, part of circle or
    #     circular ring if the cells are neighbours at the same level. If points are not neighbours, returns None.
    #     """
    #     point_1 = Point2D(index_1, self)
    #     point_2 = Point2D(index_2, self)
    #
    #     # if they are sideways neighbours
    #     if self._at_same_radius(index_1, index_2):
    #         # vertices_above
    #         vertices_1a = point_1.get_vertices_above()
    #         vertices_2a = point_2.get_vertices_above()
    #         r_larger = np.linalg.norm(vertices_1a[0])
    #         set_vertices_1a = set([tuple(v) for v in vertices_1a])
    #         set_vertices_2a = set([tuple(v) for v in vertices_2a])
    #         # vertices that are above point 1 and point 2
    #         intersection_a = set_vertices_1a.intersection(set_vertices_2a)
    #         # vertices below - only important to determine radius
    #         vertices_1b = point_1.get_vertices_below()
    #         r_smaller = np.linalg.norm(vertices_1b[0])
    #
    #         if not self._are_sideways_neighbours(index_1, index_2):
    #             if print_message:
    #                 print(f"Points {index_1} and {index_2} are not neighbours.")
    #             return None
    #         else:
    #             return r_larger - r_smaller
    #     # if point_1 right above point_2
    #     if self._point1_right_above_point2(point_1, point_2):
    #         return point_2.get_area_above()
    #     if self._point2_right_above_point1(point_1, point_2):
    #         return point_1.get_area_above()
    #     # if no exit point so far
    #     if print_message:
    #         print(f"Points {index_1} and {index_2} are not neighbours.")
    #     return None

    def get_volume(self, index: int) -> float:
        """
        Get the volume of any cell in FullVoronoiGrid, defined by its index in flattened position grid.
        """
        return self.get_all_voronoi_volumes()[index]

    def get_all_voronoi_volumes(self) -> NDArray:
        """
        Get an array in the same order as flat position grid, listing the volumes of Voronoi cells.
        """
        vor_radius = list(self.full_grid.get_between_radii())
        vor_radius.insert(0, 0)
        vor_radius = np.array(vor_radius)
        ideal_volumes = pi * (vor_radius[1:] ** 2 - vor_radius[:-1] ** 2) / self.full_grid.num_angular
        volumes = []
        for i, point in enumerate(self.full_grid.get_flattened_cartesian_coords()):
            layer = i % self.full_grid.num_radial
            volumes.append(ideal_volumes[layer])

        volumes = np.array(volumes)
        return volumes

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
        N_pos_array = len(self.flat_positions)
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

    def get_all_voronoi_surfaces(self) -> coo_array:
        """
        If l is the length of the flattened position array, returns a lxl (sparse) array where the i-th row and j-th
        column (as well as the j-th row and the i-th column) represent the size of the Voronoi surface between points
        i and j in position grid. If the points do not share a division area, no value will be set.
        """
        return self._get_property_all_pairs(self.full_grid.get_division_area)

    def get_all_distances_between_centers(self) -> coo_array:
        """
        Get a sparse matrix where for all sets of neighbouring cells the distance between Voronoi centers is provided.
        Therefore, the value at [i][j] equals the value at [j][i]. For more information, check
        self.get_distance_between_centers.
        """
        return self._get_property_all_pairs(self.full_grid.get_distance_between_centers)

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

# class Point2D:
#
#     """
#     A Point represents a single cell in a particular FullVoronoiGrid. It holds all relevant information, eg. the
#     index of the cell within a single layer and in a full flattened position grid. It enables the identification
#     of Voronoi vertices and connected calculations (distances, areas, volumes).
#     """
#
#     def __init__(self, index_position_grid: int, full_sv: FlatVoronoiGrid):
#         self.full_sv = full_sv
#         self.index_position_grid: int = index_position_grid
#         self.point: NDArray = self.full_sv.flat_positions[index_position_grid]
#         self.d_to_origin: float = np.linalg.norm(self.point)
#         self.index_radial: int = self._find_index_radial()
#         self.index_within_sphere: int = self._find_index_within_sphere()
#
#     def get_normalised_point(self) -> NDArray:
#         """Get the vector to the grid point (center of Voronoi cell) normalised to length 1."""
#         return normalise_vectors(self.point, length=1)
#
#     def _find_index_radial(self) -> int:
#         """Find to which radial layer of points this point belongs."""
#         point_radii = self.full_sv.full_grid.rs
#         for i, dist in enumerate(point_radii):
#             if np.isclose(dist, self.d_to_origin):
#                 return i
#         else:
#             raise ValueError("The norm of the point not close to any of the radii.")
#
#     def _find_index_within_sphere(self) -> int:
#         """Find the index of the point within a single layer of possible orientations."""
#         radial_index = self.index_radial
#         num_radial = len(self.full_sv.full_grid.rs)
#         return (self.index_position_grid - radial_index) // num_radial
#
#     def _find_index_sv_above(self) -> Optional[int]:
#         for i, sv in enumerate(self.full_sv.get_voronoi_discretisation()):
#             if sv.radius > self.d_to_origin:
#                 return i
#         else:
#             # the point is outside the largest voronoi sphere
#             return None
#
#     def _get_sv_above(self) -> SphericalVoronoi:
#         """Get the spherical Voronoi with the first radius that is larger than point radius."""
#         return self.full_sv.get_voronoi_discretisation()[self._find_index_sv_above()]
#
#     def _get_sv_below(self) -> Optional[SphericalVoronoi]:
#         """Get the spherical Voronoi with the largest radius that is smaller than point radius. If the point is in the
#         first layer, return None."""
#         index_above = self._find_index_sv_above()
#         if index_above != 0:
#             return self.full_sv.get_voronoi_discretisation()[index_above - 1]
#         else:
#             return None
#
#     ##################################################################################################################
#     #                            GETTERS - DISTANCES, AREAS, VOLUMES
#     ##################################################################################################################
#
#     def get_radius_above(self) -> float:
#         """Get the radius of the SphericalVoronoi cell that is the upper surface of the cell."""
#         sv_above = self._get_sv_above()
#         return sv_above.radius
#
#     def get_radius_below(self) -> float:
#         """Get the radius of the SphericalVoronoi cell that is the lower surface of the cell (return zero if there is
#         no Voronoi layer below)."""
#         sv_below = self._get_sv_below()
#         if sv_below is None:
#             return 0.0
#         else:
#             return sv_below.radius
#
#     def get_area_above(self) -> float:
#         """Get the area of the Voronoi surface that is the upper side of the cell (curved surface, part of sphere)."""
#         sv_above = self._get_sv_above()
#         areas = sv_above.calculate_areas()
#         return areas[self.index_within_sphere]
#
#     def get_area_below(self) -> float:
#         """Get the area of the Voronoi surface that is the lower side of the cell (curved surface, part of sphere)."""
#         sv_below = self._get_sv_below()
#         if sv_below is None:
#             return 0.0
#         else:
#             areas = sv_below.calculate_areas()
#             return areas[self.index_within_sphere]
#
#     def get_vertices_above(self) -> NDArray:
#         """Get the vertices of this cell that belong to the SphericalVoronoi above the point."""
#         sv_above = self._get_sv_above()
#         regions = sv_above.regions[self.index_within_sphere]
#         vertices_above = sv_above.vertices[regions]
#         return vertices_above
#
#     def get_vertices_below(self) -> NDArray:
#         """Get the vertices of this cell that belong to the SphericalVoronoi below the point (or just the origin
#         if the point belongs to the first layer."""
#         sv_below = self._get_sv_below()
#         if sv_below is None:
#             vertices_below = np.zeros((1, 3))
#         else:
#             regions = sv_below.regions[self.index_within_sphere]
#             vertices_below = sv_below.vertices[regions]
#
#         return vertices_below
#
#     def get_vertices(self) -> NDArray:
#         """Get all vertices of this cell as a single array."""
#         vertices_above = self.get_vertices_above()
#         vertices_below = self.get_vertices_below()
#
#         return np.concatenate((vertices_above, vertices_below))
