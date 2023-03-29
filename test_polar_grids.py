"""
Tests for polar_grids module.
"""
from scipy.constants import pi
import numpy as np

from polar_grids import PolarGrid


def test_polar_grid():
    my_polar_grid = PolarGrid(num_radial=3, num_angular=4)

    # assert defaults set
    assert my_polar_grid.r_lim == (0, 10)

    # correct 1D grids
    assert np.allclose(my_polar_grid.rs, [0, 5, 10])
    assert np.allclose(my_polar_grid.thetas, [0, pi/2, pi, 3*pi/2])

    # radial grid only
    expected_rad_grid = np.array([[0, 15.6], [5, 15.6], [10, 15.6]])
    real_rad_grid = my_polar_grid.get_radial_grid(theta=15.6)
    assert np.allclose(real_rad_grid, expected_rad_grid)

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
