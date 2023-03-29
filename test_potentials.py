"""
Test the potentials module.
"""

import numpy as np
from scipy.constants import pi

from potentials import FlatSymmetricalDoubleWell, AnalyticalCircularPotential


def _test_period_is_2pi(potential: AnalyticalCircularPotential):
    r_cons = 3.2
    theta_const = 0.276
    periodic_theta = np.array([[r_cons, theta_const], [r_cons, theta_const + 2*pi], [r_cons, theta_const - 2*pi],
                               [r_cons, theta_const + 7*2*pi], [r_cons, theta_const - 13*2*pi]])
    potentials = potential.get_potential(periodic_theta)
    assert np.allclose(potentials, potentials[0])


def test_flat_symmetrical_double_well():
    my_potential = FlatSymmetricalDoubleWell(10, 2, 1)
    _test_period_is_2pi(my_potential)

    list_option = my_potential.get_potential([2, 3])
    tuple_option = my_potential.get_potential((2, 3))
    array_option1 = my_potential.get_potential(np.array([2, 3]))
    array_option2 = my_potential.get_potential(np.array([[2, 3]]))

    assert np.allclose((list_option, tuple_option, array_option1), array_option2)

    assert np.isclose(my_potential.get_potential([1, 0]), 0)
    assert np.isclose(my_potential.get_potential([2, 0]), 10)
    assert np.isclose(my_potential.get_potential([0, 0]), 90)
    assert np.isclose(my_potential.get_potential([3, 0]), 0)
