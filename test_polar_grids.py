"""
Tests for polar_grids module.
"""
from scipy.constants import pi
import numpy as np

from polar_grids import PolarGrid, from_polar_to_cartesian


def test_conversion():
    """Tests simple conversion (r, theta) -> (x, y)"""
    my_rs = np.array([0, 1, 1.5, 1.2])
    my_thetas = np.array([pi/7, pi/2, 3*pi, pi/4])
    expected_xs = np.array([0, 0, -1.5, np.sqrt(2)/2 * 1.2])
    expected_ys = np.array([0, 1, 0, np.sqrt(2) / 2 * 1.2])
    conv_result = from_polar_to_cartesian(my_rs, my_thetas)
    assert np.allclose(conv_result[0], expected_xs)
    assert np.allclose(conv_result[1], expected_ys)


def test_polar_grid():
    my_polar_grid = PolarGrid(num_radial=3, num_angular=4, r_lim=(0, 10))

    # correct 1D grids
    assert np.allclose(my_polar_grid.get_rs(), [0, 5, 10])
    assert np.allclose(my_polar_grid.get_thetas(), [0, pi/2, pi, 3*pi/2])

    # radial grid only
    expected_rad_grid = np.array([[0, 15.6], [5, 15.6], [10, 15.6]])
    real_rad_grid = my_polar_grid.get_radial_grid(theta=15.6)
    assert np.allclose(real_rad_grid, expected_rad_grid)

    # angular grid only
    exp_unit_angular = np.array([[1, 0], [1, pi/2], [1, pi], [1, 3*pi/2]])
    assert np.allclose(my_polar_grid.get_unit_angular_grid(), exp_unit_angular)

    # shape and content of polar meshgrids

    expected_meshgrid_r = np.array([
        [0, 5, 10],
        [0, 5, 10],
        [0, 5, 10],
        [0, 5, 10]
    ])
    expected_meshgrid_theta = np.array([
        [0, 0, 0],
        [pi/2, pi/2, pi/2],
        [pi, pi, pi],
        [3*pi/2, 3*pi/2, 3*pi/2]
    ])

    real_meshgrid = my_polar_grid.get_polar_meshgrid()
    assert np.allclose(real_meshgrid[0], expected_meshgrid_r)
    assert np.allclose(real_meshgrid[1], expected_meshgrid_theta)

    # checking x and y grids via relation x^2 + y^2 = r^2
    x_mg, y_mg = my_polar_grid.get_cartesian_meshgrid()
    assert np.allclose(x_mg**2 + y_mg**2, expected_meshgrid_r**2)

    # flattened grids
    exp_flat_polar = np.array([[0, 0], [5, 0], [10, 0], [0, pi/2], [5, pi/2], [10, pi/2], [0, pi], [5, pi], [10, pi],
                               [0, 3*pi/2], [5, 3*pi/2], [10, 3*pi/2]])

    assert np.allclose(my_polar_grid.get_flattened_polar_coords(), exp_flat_polar)

    rs_flat = exp_flat_polar.T[0]

    flat_cartesian = my_polar_grid.get_flattened_cartesian_coords()
    xs_flat = flat_cartesian.T[0]
    ys_flat = flat_cartesian.T[1]

    assert np.allclose(xs_flat**2 + ys_flat**2, rs_flat**2)

    # a larger grid with different limits
    num_rad = 22
    num_ang = 37
    my_polar_grid2 = PolarGrid(r_lim=(3, 8), num_radial=num_rad, num_angular=num_ang)

    # correct 1D grids
    expected_rs2 = [3., 3.23809524, 3.47619048, 3.71428571, 3.95238095, 4.19047619, 4.42857143, 4.66666667, 4.9047619,
                    5.14285714, 5.38095238, 5.61904762, 5.85714286, 6.0952381,  6.33333333, 6.57142857, 6.80952381,
                    7.04761905, 7.28571429, 7.52380952, 7.76190476, 8.]
    expected_thetas2 = [0., 0.16981582, 0.33963164, 0.50944746, 0.67926328, 0.8490791, 1.01889491, 1.18871073,
                        1.35852655, 1.52834237, 1.69815819, 1.86797401, 2.03778983, 2.20760565, 2.37742147, 2.54723729,
                        2.71705311, 2.88686892, 3.05668474, 3.22650056, 3.39631638, 3.5661322,  3.73594802, 3.90576384,
                        4.07557966, 4.24539548, 4.4152113,  4.58502712, 4.75484294, 4.92465875, 5.09447457, 5.26429039,
                        5.43410621, 5.60392203, 5.77373785, 5.94355367, 6.11336949]

    assert np.allclose(my_polar_grid2.get_rs(), expected_rs2)
    assert np.allclose(my_polar_grid2.get_thetas(), expected_thetas2)

    # unit cartesian grid
    # expect that x^2 + y^2 = 1
    x_uc, y_uc = my_polar_grid2.get_unit_cartesian_grid().T
    assert np.allclose(x_uc ** 2 + y_uc ** 2, 1)

    polar_mg = my_polar_grid2.get_polar_meshgrid()
    cartesian_mg = my_polar_grid2.get_cartesian_meshgrid()
    assert polar_mg[0].shape == (num_ang, num_rad)
    assert polar_mg[1].shape == (num_ang, num_rad)
    assert cartesian_mg[0].shape == (num_ang, num_rad)
    assert cartesian_mg[1].shape == (num_ang, num_rad)
    assert my_polar_grid2.get_radial_grid(theta=7).shape == (num_rad, 2)
    flat_polar = my_polar_grid2.get_flattened_polar_coords()
    flat_car = my_polar_grid2.get_flattened_cartesian_coords()
    assert flat_polar.shape == (num_ang*num_rad, 2)
    assert flat_car.shape == (num_ang * num_rad, 2)

    # coordinate transformations
    xs = flat_car.T[0]
    ys = flat_car.T[1]
    rs = flat_polar.T[0]
    assert np.allclose(xs**2 + ys**2, rs**2)


def test_neighbour_relationships():
    num_rad = 5
    num_ang = 8
    my_polar_grid = PolarGrid(r_lim=(3, 8), num_radial=num_rad, num_angular=num_ang)

    # layer indices
    assert my_polar_grid._index_to_layer(0) == 0
    assert my_polar_grid._index_to_layer(1) == 1
    assert my_polar_grid._index_to_layer(7) == 2
    assert my_polar_grid._index_to_layer(5*8-1) == 4
    assert np.isnan(my_polar_grid._index_to_layer(5 * 8))

    # slice indices
    assert my_polar_grid._index_to_slice(0) == 0
    assert my_polar_grid._index_to_slice(4) == 0
    assert my_polar_grid._index_to_slice(5) == 1
    assert my_polar_grid._index_to_slice(9) == 1
    assert my_polar_grid._index_to_slice(5 * 8 - 1) == 7
    assert np.isnan(my_polar_grid._index_to_slice(5 * 8))

    # side neighbours
    assert my_polar_grid._are_sideways_neighbours(0, 5)
    assert my_polar_grid._are_sideways_neighbours(0, 35)
    assert my_polar_grid._are_sideways_neighbours(4, 39)
    assert my_polar_grid._are_sideways_neighbours(6, 11)
    assert not my_polar_grid._are_sideways_neighbours(0, 1)
    assert not my_polar_grid._are_sideways_neighbours(0, 10)
    assert not my_polar_grid._are_sideways_neighbours(9, 10)
    assert not my_polar_grid._are_sideways_neighbours(0, 30)

    # ray neighbours
    assert my_polar_grid._are_ray_neighbours(0, 1)
    assert my_polar_grid._are_ray_neighbours(1, 0)
    assert my_polar_grid._are_ray_neighbours(3, 4)
    assert my_polar_grid._are_ray_neighbours(35, 36)
    assert my_polar_grid._are_ray_neighbours(32, 31)
    assert not my_polar_grid._are_ray_neighbours(30, 35)
    assert not my_polar_grid._are_ray_neighbours(30, 32)
    assert not my_polar_grid._are_ray_neighbours(39, 3)
    assert not my_polar_grid._are_ray_neighbours(39, 4)


def test_distances():
    my_pg = PolarGrid(r_lim=(3, 7), num_radial=5, num_angular=15)

    dist_array = my_pg.get_all_distances_between_centers_as_numpy()

    # assert is diagonal
    assert np.allclose(dist_array, dist_array.T, equal_nan=True)

    # some points on same ray
    assert np.isclose(dist_array[0][1], 1)
    assert np.isclose(dist_array[3][4], 1)
    assert np.isclose(dist_array[67][68], 1)
    assert np.isclose(dist_array[50][51], 1)
    assert np.isclose(dist_array[73][74], 1)

    # some points at same distance
    # radius of the smallest circle is 3, distance is 2*pi*3/15
    assert np.isclose(dist_array[0][5], 2*pi*3/15)
    assert np.isclose(dist_array[0][70], 2*pi*3/15)
    # radius of the second circle is 4, distance is 2*pi*4/15
    assert np.isclose(dist_array[1][6], 2 * pi * 4 / 15)
    assert np.isclose(dist_array[1][71], 2 * pi * 4 / 15)
    assert np.isclose(dist_array[26][31], 2 * pi * 4 / 15)
    # and the last circle with r=7
    assert np.isclose(dist_array[4][9], 2 * pi * 7 / 15)
    assert np.isclose(dist_array[4][74], 2 * pi * 7 / 15)
    assert np.isclose(dist_array[24][29], 2 * pi * 7 / 15)

    # some points that are not neighbours
    assert np.isnan(dist_array[0][2])
    assert np.isnan(dist_array[0][4])
    assert np.isnan(dist_array[70][74])
    assert np.isnan(dist_array[69][66])
    assert np.isnan(dist_array[69][70])
    assert np.isnan(dist_array[0][10])
    assert np.isnan(dist_array[0][71])
    assert np.isnan(dist_array[0][65])
    assert np.isnan(dist_array[4][73])
    assert np.isnan(dist_array[0][10])
    assert np.isnan(dist_array[32][42])
    assert np.isnan(dist_array[32][41])
    assert np.isnan(dist_array[30][34])
    assert np.isnan(dist_array[6][70])
    assert np.isnan(dist_array[6][71])
    assert np.isnan(dist_array[54][64])


def test_areas():
    my_pg = PolarGrid(r_lim=(3, 7), num_radial=5, num_angular=15)
    surf_array = my_pg.get_all_voronoi_surfaces_as_numpy()

    # assert is diagonal
    assert np.allclose(surf_array, surf_array.T, equal_nan=True)

    # some points on same ray
    # radius of the smallest division circle is 3.5, distance is 2*pi*3.5/15
    assert np.isclose(surf_array[0][1], 2*pi*3.5/15)
    assert np.isclose(surf_array[5][6], 2 * pi * 3.5 / 15)
    assert np.isclose(surf_array[70][71], 2 * pi * 3.5 / 15)
    # second circle
    assert np.isclose(surf_array[1][2], 2 * pi * 4.5 / 15)
    assert np.isclose(surf_array[6][7], 2 * pi * 4.5 / 15)
    assert np.isclose(surf_array[71][72], 2 * pi * 4.5 / 15)
    # largest circle
    assert np.isclose(surf_array[3][4], 2 * pi * 6.5 / 15)
    assert np.isclose(surf_array[8][9], 2 * pi * 6.5 / 15)
    assert np.isclose(surf_array[73][74], 2 * pi * 6.5 / 15)

    # some points at same distance
    assert np.isclose(surf_array[0][5], 3.5)
    assert np.isclose(surf_array[0][70], 3.5)
    # except for the inner circle side areas are 1
    assert np.isclose(surf_array[1][6], 1)
    assert np.isclose(surf_array[1][71], 1)
    assert np.isclose(surf_array[26][31], 1)
    # and the last circle with r=7
    assert np.isclose(surf_array[4][9], 1)
    assert np.isclose(surf_array[4][74], 1)
    assert np.isclose(surf_array[24][29], 1)

    # some points that are not neighbours
    assert np.isnan(surf_array[0][2])
    assert np.isnan(surf_array[0][4])
    assert np.isnan(surf_array[70][74])
    assert np.isnan(surf_array[69][66])
    assert np.isnan(surf_array[69][70])
    assert np.isnan(surf_array[0][10])
    assert np.isnan(surf_array[0][71])
    assert np.isnan(surf_array[0][65])
    assert np.isnan(surf_array[4][73])
    assert np.isnan(surf_array[0][10])
    assert np.isnan(surf_array[32][42])
    assert np.isnan(surf_array[32][41])
    assert np.isnan(surf_array[30][34])
    assert np.isnan(surf_array[6][70])
    assert np.isnan(surf_array[6][71])
    assert np.isnan(surf_array[54][64])


def test_volumes():
    my_pg = PolarGrid(r_lim=(3, 7), num_radial=5, num_angular=15)
    voronoi_volumes = my_pg.get_all_voronoi_volumes()

    # first layer
    assert np.allclose(voronoi_volumes[0:74:5], pi*3.5**2/15)
    # higher layers where you need to subtract previous ones
    assert np.allclose(voronoi_volumes[1:74:5], pi * 4.5 ** 2 / 15 - pi * 3.5 ** 2 / 15)
    assert np.allclose(voronoi_volumes[2:74:5], pi * 5.5 ** 2 / 15 - pi * 4.5 ** 2 / 15)
    assert np.allclose(voronoi_volumes[3:74:5], pi * 6.5 ** 2 / 15 - pi * 5.5 ** 2 / 15)
    assert np.allclose(voronoi_volumes[4:74:5], pi * 7.5 ** 2 / 15 - pi * 6.5 ** 2 / 15)


def test_layers_and_slices():
    my_pg = PolarGrid(r_lim=(2, 18), num_radial=3, num_angular=5)
    indices = list(range(3*5))

    # all layer indices
    layer_indices = []
    for i in indices:
        layer_indices.append(my_pg._index_to_layer(i))
    assert layer_indices == [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]

    # all slices indices
    slices_indices = []
    for i in indices:
        slices_indices.append(my_pg._index_to_slice(i))
    assert slices_indices == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]

    # all indices per layer
    all_0_layer = my_pg.get_all_indices_of_layer_i(0)
    assert all_0_layer == [0, 3, 6, 9, 12]
    all_1_layer = my_pg.get_all_indices_of_layer_i(1)
    assert all_1_layer == [1, 4, 7, 10, 13]
    all_2_layer = my_pg.get_all_indices_of_layer_i(2)
    assert all_2_layer == [2, 5, 8, 11, 14]

    # all indices per slice
    all_0_slice = my_pg.get_all_indices_of_slice_j(0)
    assert all_0_slice == [0, 1, 2]
    all_1_slice = my_pg.get_all_indices_of_slice_j(1)
    assert all_1_slice == [3, 4, 5]
    all_last_slice = my_pg.get_all_indices_of_slice_j(4)
    assert all_last_slice == [12, 13, 14]


def test_pairwise_properties():
    my_pg = PolarGrid(r_lim=(2, 18), num_radial=3, num_angular=5)
    print(my_pg._get_property_all_pairs(my_pg._test_pair_property))


if __name__ == "__main__":
    test_pairwise_properties()