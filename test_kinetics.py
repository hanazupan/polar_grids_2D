import numpy as np
import scipy
from scipy.constants import R

from kinetics import FlatSQRA


four_cell_neighbours = np.array([[np.nan, 1, 1, np.nan],
                                 [1, np.nan, np.nan, 1],
                                 [1, np.nan, np.nan, 1],
                                 [np.nan, 1, 1, np.nan]])


class TestKinModel1(FlatSQRA):

    """
    Create a mini test model of square cells with potentials

    -------------
    | 0.5 |  1  |
    |  1  | 0.3 |
    -------------

    cells are numbered

    -------------
    | 0 | 1 |
    | 2 | 3 |
    -------------

    and are squares with side 1.

    """

    def __init__(self, **kwargs):
        super().__init__(discretisation_grid=None, potential=None, **kwargs)

    def get_volumes(self):
        return np.full((4,), 1)

    def get_num_of_cells(self):
        return 4

    def get_surface_areas(self):
        return four_cell_neighbours

    def get_center_distances(self):
        return four_cell_neighbours

    def get_potentials(self):
        return np.array([0.5, 1, 1, 0.3])

    def get_populations(self):
        not_norm_pop = np.exp(-self.get_potentials() / (R * self.T))
        pop_sum = np.sum(not_norm_pop)
        return 1 / pop_sum * not_norm_pop

def get_non_sparse_eigv(kin_model):
    trans_matrix = kin_model.get_transition_matrix()
    eigenval, left_eigenvec = scipy.linalg.eig(trans_matrix, left=True, right=False)
    # sort
    idx = eigenval.argsort()[::-1]
    eigenval = eigenval[idx]
    left_eigenvec = left_eigenvec[:, idx]
    # make real if possible
    if left_eigenvec.imag.max() == 0 and eigenval.imag.max() == 0:
        left_eigenvec = left_eigenvec.real
        eigenval = eigenval.real
    return eigenval, left_eigenvec


def _test_kinetic_model(my_model, num_cells):
    """
    Perform basic tests on the kinetic model of SqRA:
        1) eigenvalues: first zero, then all below zero
        2) eigenvectors: first no sign change, second one sign change ...
    Args:
        my_model:
        num_cells:

    Returns:

    """
    eigval, eigvec = my_model.get_eigenval_eigenvec(num_cells//2)

    non_sparse_eigval, non_sparse_eigenvec = get_non_sparse_eigv(my_model)

    # rate matrix is 0 for ij where i=/=j and i and j are not neighbours
    rate_matrix = my_model.get_transition_matrix()
    neighbours = ~np.isnan(my_model.get_surface_areas())
    np.fill_diagonal(neighbours, True)
    # fields that are not neighbours or diagonals must be zero
    assert np.allclose(rate_matrix[~neighbours], 0)
    # row sum of rate matrix must be zero
    for row in rate_matrix:
        assert np.allclose(np.sum(row), 0)

    # first eigenvalue close to zero
    assert np.isclose(eigval[0], 0)
    # subsequent eigenvalues all negative
    assert np.all(eigval[1:] < 0)
    assert np.all(non_sparse_eigval[1:] < 0)
    # all eigenval equal to non sparse eigenval
    assert np.allclose(eigval[1], non_sparse_eigval[1])

    # first eigenvector - all positive or all negative
    assert np.allclose(np.sign(eigvec.T[0]), 1) or np.allclose(np.sign(eigvec.T[0]), -1)

    # i-th eigenvector (starting with 0) has i sign changes
    for i in range(num_cells//2):
        num_changes = (np.diff(np.sign(eigvec.T[i])) != 0)*1
        assert np.count_nonzero(num_changes) == i

    # eigenvectors the same as non-sparse ones (up to a sign)
    for i in range(num_cells//2):
        assert np.allclose(eigvec.T[i], non_sparse_eigenvec.T[i]) or np.allclose(eigvec.T[i], -non_sparse_eigenvec.T[i])


def test_model1():
    m1 = TestKinModel1()
    _test_kinetic_model(m1, 4)
    # because cells 1 and 2 have same potentials and shapes, the value of eigenvector there should be the same
    eigval, eigvec = m1.get_eigenval_eigenvec(2)
    assert np.isclose(eigvec.T[0][1], eigvec.T[0][2])


if __name__ == "__main__":
    test_model1()
