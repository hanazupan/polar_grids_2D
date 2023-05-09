from typing import Tuple

import numpy as np
import scipy
from numpy.typing import NDArray
from scipy.constants import R
from scipy.sparse.linalg import eigs, eigsh

from polar_grids import PolarGrid
from potentials import AnalyticalCircularPotential


class FlatSQRA:

    def __init__(self, discretisation_grid: PolarGrid, potential: AnalyticalCircularPotential,
                 D: float = 1.0, T: float = 300.0):
        self.discretisation_grid = discretisation_grid
        self.potential = potential
        self.D = D
        self.T = T
        self.transition_matrix = None
        self.eigenval = np.empty((1,))
        self.eigenvec = None

    def get_volumes(self):
        return self.discretisation_grid.get_full_voronoi_grid().get_all_voronoi_volumes()

    def get_num_of_cells(self):
        return self.discretisation_grid.num_angular * self.discretisation_grid.num_radial

    def get_surface_areas(self):
        return self.discretisation_grid.get_full_voronoi_grid().get_all_voronoi_surfaces_as_numpy()

    def get_center_distances(self):
        return self.discretisation_grid.get_full_voronoi_grid().get_all_distances_between_centers_as_numpy()

    def get_potentials(self):
        return self.potential.get_potential(self.discretisation_grid.get_flattened_polar_coords())

    def get_populations(self):
        return self.potential.get_population(self.discretisation_grid.get_flattened_polar_coords(), T=self.T)

    def get_transition_matrix(self):
        """
        Apply the formula:

        Q_ij = D*S_ij/(h_i*V_i) * e^[(V_i-V_j)/(2*R*T)]
        Units within the exponent: V [kJ/mol], RT [kj/mol]
        """
        if self.transition_matrix is None:
            surface_areas = self.get_surface_areas()
            distances = self.get_center_distances()
            volumes = self.get_volumes()
            potentials = self.get_potentials()
            self.transition_matrix = np.divide(self.D * surface_areas, distances)
            for i, _ in enumerate(self.transition_matrix):
                self.transition_matrix[i] = np.divide(self.transition_matrix[i], volumes[i],
                                                      out=np.zeros_like(self.transition_matrix[i]))
            for j, _ in enumerate(self.transition_matrix):
                for i, _ in enumerate(self.transition_matrix):
                    # only for neighbours
                    if surface_areas[i, j]:
                        self.transition_matrix[i, j] *= np.exp((potentials[i]-potentials[j])*1000/(2*R*self.T))
            self.transition_matrix[np.isnan(self.transition_matrix)] = 0.0
            # normalise rows
            sums = np.sum(self.transition_matrix, axis=1)
            np.fill_diagonal(self.transition_matrix, -sums)
        return self.transition_matrix

    def get_eigenval_eigenvec(self, num_eigenv: int = 15, **kwargs) -> Tuple[NDArray, NDArray]:
        """
        Obtain eigenvectors and eigenvalues of the transition matrices.

        Args:
            num_eigenv: how many eigenvalues/vectors pairs to return (too many may give inaccurate results)
            **kwargs: named arguments to forward to eigs()
        Returns:
            (eigenval, eigenvec) a tuple of eigenvalues and eigenvectors, first num_eigv given for all tau-s
            Eigenval is of shape (num_tau, num_eigenv), eigenvec of shape (num_tau, num_cells, num_eigenv)
        """
        if self.eigenvec is None or self.eigenval is None or len(self.eigenval) < num_eigenv:
            tm = self.get_transition_matrix()
            # in order to compute left eigenvectors, compute right eigenvectors of the transpose
            eigenval, eigenvec = eigs(tm.T, num_eigenv, maxiter=100000, which="SM", tol=0, **kwargs) #,
            #eigenval = 1/(eigenval - sigma)
            # don't need to deal with complex outputs in case all values are real
            if eigenvec.imag.max() == 0 and eigenval.imag.max() == 0:
                eigenvec = eigenvec.real
                eigenval = eigenval.real
            # sort eigenvectors according to their eigenvalues
            idx = eigenval.argsort()[::-1]
            self.eigenval = eigenval[idx]
            self.eigenvec = eigenvec[:, idx]
            # normalise so that the first value of eigenvec array that is not zero is positive
            index_first_values = np.argmax(np.abs(self.eigenvec), axis=0)
            for i, row in enumerate(self.eigenvec.T):
                self.eigenvec[:, i] *= - np.sign(row[index_first_values[i]])
        return self.eigenval, self.eigenvec

    def get_its(self, num_eigenv: int = 15, **kwargs):
        eigenvals, eigenvecs = self.get_eigenval_eigenvec(num_eigenv=num_eigenv, **kwargs)
        eigenvals = np.array(eigenvals)
        its = - 1 / eigenvals[1:]
        return its


if __name__ == "__main__":
    from potentials import FlatSymmetricalDoubleWell
    potential = FlatSymmetricalDoubleWell(10, 2, 1)
    pg = PolarGrid(r_lim=(3, 7), num_radial=50, num_angular=15)
    my_model = FlatSQRA(pg, potential)
    print(my_model.get_transition_matrix())

