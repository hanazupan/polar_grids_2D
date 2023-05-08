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

def test_model1():
    m1 = TestKinModel1()
    eigval, eigvec = m1.get_eigenval_eigenvec(2)

    non_sparse_eigval, non_sparse_eigenvec = get_non_sparse_eigv(m1)
    # first eigenvalue close to zero
    assert np.isclose(eigval[0], 0)
    # subsequent eigenvalues all negative
    assert np.all(eigval[1:] < 0)
    assert np.all(non_sparse_eigval[1:] < 0)
    # all eigenval equal to non sparse eigenval
    assert np.allclose(eigval[1], non_sparse_eigval[1])

    # first eigenvector - all positive or all negative
    assert np.allclose(np.sign(eigvec.T[0]), 1) or np.allclose(np.sign(eigvec.T[0]), -1)
    print(eigvec.T[0], non_sparse_eigenvec.T[0])
    print(m1.get_its(2))

if __name__ == "__main__":
    test_model1()
