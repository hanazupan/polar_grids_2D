"""
Plot potentials here.
"""
from functools import wraps
from typing import Union, Callable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import colors
from matplotlib.animation import ArtistAnimation, FuncAnimation
from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure

from kinetics import FlatSQRA
from molgri.constants import DIM_SQUARE
from molgri.plotting.abstract import RepresentationCollection
from molgri.space.utils import normalise_vectors
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import geometric_slerp
import matplotlib.pylab as pl

from potentials import AnalyticalCircularPotential, FlatDoubleWellAlpha, RadialMinDoubleWellAlpha, \
    FlatSymmetricalDoubleWell
from polar_grids import PolarGrid


#######################################################################################################################
#                                                 DECORATORS
#######################################################################################################################

def fig_ax_wrapper(my_method):

    """
    This wrapper deals with creation and saving of figures. It is added to any method that plots.
    """

    @wraps(my_method)
    def decorated(*args, ax: Union[Axes, Axes3D] = None, fig: Figure = None, save: bool = True,
                  creation_kwargs: dict = None, data_name: str = None, **kwargs):
        """
        If you already have ax and/or fig you would like to add to, specify it in the argument list. If you don't
        want to immediately save the figure or want to specify some other generation details, also provide the
        necessary arguments here.
        """
        self = args[0]
        if creation_kwargs is None:
            creation_kwargs = dict()
        # pre-processing (creating fig, ax if needed)
        self._create_fig_ax(fig=fig, ax=ax, **creation_kwargs)
        # calling the function
        func_value = my_method(*args, **kwargs)
        # post-processing (saving the results)
        method_name = my_method.__name__
        if data_name:
            method_name += f"_{data_name}"
        if save:
            self._save_plot_type(method_name)
        return func_value
    return decorated


def is_3d_plot(my_method):

    """
    A 3d plot has specified projection 3d and is able to perform animation of figure rotation.
    Use before the fig_ax_wrapper
    """
    def decorated(*args, animate_rot: bool = False, creation_kwargs: dict = None, **kwargs):
        """
        To 3D plots, you can additionally provide the argument animate_rot. If True, an animation will be created and
        saved.
        """
        self = args[0]
        if creation_kwargs is None:
            creation_kwargs = dict()
        creation_kwargs["projection"] = "3d"
        func_value = my_method(*args, creation_kwargs=creation_kwargs, **kwargs)
        method_name = my_method.__name__
        if animate_rot:
            return self._animate_figure_view(self.fig, self.ax, method_name)
        return func_value
    return decorated

#######################################################################################################################
#                                        GENERAL PLOTTING METHODS
#######################################################################################################################


def plot_voronoi_cells(sv, ax, plot_vertex_points=True):
    t_vals = np.linspace(0, 1, 2000)
    # plot Voronoi vertices
    if plot_vertex_points:
        ax.scatter(sv.vertices[:, 0], sv.vertices[:, 1], c='g')
    # indicate Voronoi regions (as Euclidean polygons)
    for region in sv.regions:
        n = len(region)
        for j in range(n):
            start = sv.vertices[region][j]
            end = sv.vertices[region][(j + 1) % n]
            norm = np.linalg.norm(start)
            result = geometric_slerp(normalise_vectors(start), normalise_vectors(end), t_vals)
            ax.plot(norm * result[..., 0], norm * result[..., 1], c='k')


def _ray_plot(pg: PolarGrid, ys_method: Callable, ax, theta: float, plot_line=True, plot_scatter=False,
              title=False, **kwargs):
    coords = pg.get_radial_grid(theta=theta)
    y_values = ys_method(coords)

    color = kwargs.pop("color", "black")
    if plot_line:
        ray = sns.lineplot(x=coords.T[0], y=y_values, ax=ax, color=color, **kwargs)
    if plot_scatter:
        ray = sns.scatterplot(x=coords.T[0], y=y_values, ax=ax, color=color, **kwargs)
    if title:
        ax.set_title(r'$\theta=$' + f"{theta}")
    return ray


#######################################################################################################################
#                                                 PLOTTING CLASSES
#######################################################################################################################

class PotentialPlot(RepresentationCollection):

    """
    This is a collection of plotting methods centered around the AnalyticalCircularPotential object.
    """

    def __init__(self, acp: AnalyticalCircularPotential, grid: PolarGrid, *args, **kwargs):
        super().__init__("potential", *args, **kwargs)
        self.acp = acp
        self.grid = grid
        self.default_cmap = "coolwarm"

    def _add_colorbar(self, cmap: Union[Colormap, str], min_value: float = None, max_value: float = None,
                      norm: Normalize = None):
        if norm is None:
            norm = Normalize(min_value, max_value)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb_ax = self.fig.add_axes([1.05, .124, .04, .754])
        return self.fig.colorbar(sm, cax=cb_ax, fraction=0.046, pad=0.04)

    def _colorbar_x_y_potential(self, potentials: NDArray, cbar_label=None):
        # colorbar instead of a legend
        cbar = self._add_colorbar(cmap=self.default_cmap, min_value=np.min(potentials), max_value=np.max(potentials))
        if self.ax.get_legend():
            self.ax.get_legend().remove()

        # labels
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        cbar.ax.set_ylabel(cbar_label)

    # ##################################### PLOTS THAT ARE JUST LINES ####################################

    @fig_ax_wrapper
    def plot_potential_ray(self, theta: float = 0, **kwargs):
        """
        A plot that shows how potential changes with radial distance for a specific angle theta.
        """
        color = kwargs.pop("color", "black")
        return _ray_plot(pg=self.grid, ys_method=self.acp.get_potential, ax=self.ax, theta=theta, color=color, **kwargs)

    @fig_ax_wrapper
    def plot_population_ray(self, theta: float = 0, **kwargs):
        """
        A plot that shows how population changes with radial distance for a specific angle theta.
        """
        color = kwargs.pop("color", "blue")
        return _ray_plot(pg=self.grid, ys_method=self.acp.get_population, ax=self.ax, theta=theta, color=color, **kwargs)

    @fig_ax_wrapper
    def plot_populations_by_assignment(self):
        condition = [lambda r, t: r < 2, lambda r, t: r >= 2]
        names = ["r < 2", "r > 2"]
        colors = ["red", "blue"]

        population = self.acp.get_population(self.grid.get_flattened_polar_coords())
        assignments = self.grid.assign_to_states(condition)
        for assig, name, color in zip(assignments, names, colors):
            self.ax.hlines(np.sum(population[assig]), 0, 1, color=color, label=name)
        self.ax.set_ylim(0, 1)
        self.ax.legend()

    # ############################ PLOTS THAT ARE FLAT CIRCLES WITH COLORS IF NEEDED ###############################

    def _plot_circle(self, y_method: Callable, colorbar=False, colorbar_label=None, **kwargs):
        coords = self.grid.get_flattened_polar_coords()
        potentials = y_method(coords)
        xy_coords = self.grid.get_flattened_cartesian_coords()
        s = kwargs.pop("s", 2)
        sns.scatterplot(x=xy_coords.T[0], y=xy_coords.T[1], ax=self.ax, hue=potentials, palette=self.default_cmap, s=s,
                        legend=False)
        self._equalize_axes()
        if colorbar:
            self._colorbar_x_y_potential(potentials, cbar_label=colorbar_label)

    @fig_ax_wrapper
    def plot_potential_circles(self, colorbar=False, **kwargs):
        """
        A plot that shows the view from above - concentrical circles of potential values.
        """
        self._plot_circle(y_method=self.acp.get_potential, colorbar=colorbar, colorbar_label="Potential [kJ/mol]",
                          **kwargs)

    @fig_ax_wrapper
    def plot_population_circles(self, colorbar=False, **kwargs):
        """
        A plot that shows the view from above - concentrical circles of potential values.
        """
        self._plot_circle(y_method=self.acp.get_population, colorbar=colorbar, colorbar_label="Population", **kwargs)

    # #################################### PLOTS THAT ARE IN 3D ########################################

    def _plot_in_3D(self, ys_method: Callable, colorbar: bool, **kwargs):
        coord_meshgrid = self.grid.get_polar_meshgrid()
        X, Y = self.grid.get_cartesian_meshgrid()
        potentials = ys_method(coord_meshgrid[0], coord_meshgrid[1])

        self.ax.plot_surface(X, Y, potentials, cmap=self.default_cmap, linewidth=0, antialiased=False, alpha=0.5,
                             **kwargs)
        if colorbar:
            self._colorbar_x_y_potential(potentials)

    @is_3d_plot
    @fig_ax_wrapper
    def plot_potential_3D(self, colorbar=False, **kwargs):
        self._plot_in_3D(ys_method=self.acp.get_potential_as_meshgrid, colorbar=colorbar, **kwargs)

    @is_3d_plot
    @fig_ax_wrapper
    def plot_population_3D(self, colorbar=False, **kwargs):
        self._plot_in_3D(ys_method=self.acp.get_population_as_meshgrid, colorbar=colorbar, **kwargs)


class PolarPlot(RepresentationCollection):

    def __init__(self, pg: PolarGrid, *args, **kwargs):
        self.pg = pg
        super().__init__("polar_grid", *args, **kwargs)

    @fig_ax_wrapper
    def plot_gridpoints(self, c = "black", **kwargs):
        points = self.pg.get_flattened_cartesian_coords()
        cmap = "bwr"
        norm = kwargs.pop("norm", colors.TwoSlopeNorm(vcenter=0))
        self.ax.scatter(*points.T, c=c, cmap=cmap, norm=norm, **kwargs)

    @fig_ax_wrapper
    def plot_radial_grid(self, theta: float = 0, **kwargs):
        """
        A plot that shows how potential changes with radial distance for a specific angle theta.
        """
        _ray_plot(pg=self.pg, ax=self.ax, ys_method=(lambda x: 0), theta=theta,
                  color="black", plot_line=True, plot_scatter=True)

    @fig_ax_wrapper
    def plot_voronoi_cells(self, numbered=False, plot_gridpoints=True, plot_vertex_points=True):
        origin = np.zeros((2,))

        if numbered:
            points = self.pg.get_flattened_cartesian_coords()
            for i, point in enumerate(points):
                self.ax.text(*point, s=f"{i}")

        if plot_gridpoints:
            self.plot_gridpoints(ax=self.ax, fig=self.fig, save=False)

        voronoi_disc = self.pg.get_full_voronoi_grid().get_voronoi_discretisation()

        for i, sv in enumerate(voronoi_disc):
            plot_voronoi_cells(sv, self.ax, plot_vertex_points=plot_vertex_points)
            # plot rays from origin to highest level
            if i == len(voronoi_disc) - 1:
                for vertex in sv.vertices:
                    ray_line = np.concatenate((origin[:, np.newaxis], vertex[:, np.newaxis]), axis=1)
                    self.ax.plot(*ray_line, color="black")


class ArrayPlot(RepresentationCollection):

    """
    A tool for plotting arrays, eg by highlighting high and low values
    """

    def __init__(self, my_array: NDArray, *args, **kwargs):
        self.array = my_array
        super().__init__("array", *args, **kwargs)

    @fig_ax_wrapper
    def make_heatmap_plot(self):
        """
        This method draws the array and colors the fields according to their values (red = very large,
        blue = very small). Zero values are always white, negative ones always blue, positive ones always red.
        """
        if np.all(self.array < 0):
            cmap = "Blues"
            norm = None
        elif np.all(self.array > 0):
            cmap = "Reds"
            norm = None
        else:
            cmap = "bwr"
            norm = colors.TwoSlopeNorm(vcenter=0)
        sns.heatmap(self.array, cmap=cmap, ax=self.ax, xticklabels=False, yticklabels=False, norm=norm)

        self._equalize_axes()


class KineticsPlot(RepresentationCollection):

    def __init__(self, kinetics_model: FlatSQRA):
        super().__init__("kinetics")
        self.kinetics_model = kinetics_model

    @fig_ax_wrapper
    def make_its_plot(self, num_eigenv=6):
        """
        Plot iterative timescales.
        """
        eigenvals, eigenvecs = self.kinetics_model.get_eigenval_eigenvec(num_eigenv=num_eigenv)
        eigenvals = np.array(eigenvals)

        # for SQRA plot vertical lines
        for j in range(1, num_eigenv):
            x_min, x_max = self.ax.get_xlim()
            self.ax.hlines(- 1 / eigenvals[j], 0, 1, color="black", ls="--")

        self.ax.set_xlim(left=0, right=1)
        self.ax.set_xlabel(r"$\tau$")
        self.ax.set_ylabel(r"ITS")

    @fig_ax_wrapper
    def make_eigenvalues_plot(self, num_eigenv=6):
        """
        Visualize the eigenvalues of rate matrix.
        """

        eigenvals, eigenvecs = self.kinetics_model.get_eigenval_eigenvec(num_eigenv=num_eigenv)

        xs = np.linspace(0, 1, num=len(eigenvals))
        self.ax.scatter(xs, eigenvals, s=5, c="black")
        for i, eigenw in enumerate(eigenvals):
            self.ax.vlines(xs[i], eigenw, 0, linewidth=0.5, color="black")
        self.ax.hlines(0, 0, 1, color="black")
        self.ax.set_ylabel(f"Eigenvalues")
        self.ax.axes.get_xaxis().set_visible(False)

    @fig_ax_wrapper
    def make_eigenvectors_plot(self, num_eigenv: int = 5):
        """
        Visualize the energy surface and the first num (default=3) eigenvectors
        """
        self.fig, self.ax = plt.subplots(1, num_eigenv, figsize=(num_eigenv*DIM_SQUARE[0], DIM_SQUARE[0]))

        for i, subax in enumerate(self.ax.ravel()):
            self.make_one_eigenvector_plot(i, ax=subax, fig=self.fig, save=False)

    @fig_ax_wrapper
    def make_one_eigenvector_plot(self, eigenvec_index: int, **kwargs):
        eigenvals, eigenvecs = self.kinetics_model.get_eigenval_eigenvec(num_eigenv=eigenvec_index+2)

        # shape: (number_cells, num_eigenvectors)

        fgp = PolarPlot(self.kinetics_model.discretisation_grid)
        #fgp.plot_voronoi_cells(fig=self.fig, ax=self.ax, plot_gridpoints=False, numbered=False, save=False,
        #                       plot_vertex_points=False)
        fgp.plot_gridpoints(ax=self.ax, fig=self.fig, save=False, c=eigenvecs[:, eigenvec_index], **kwargs)
        self.ax.set_title(f"Eigenv. {eigenvec_index}")
        self._equalize_axes()
        #self.fig.colorbar()

    @fig_ax_wrapper
    def make_one_eigenvector_ray_plot(self, eigenvec_index: int, angular_index: int = 0, **kwargs):
        eigenvals, eigenvecs = self.kinetics_model.get_eigenval_eigenvec(num_eigenv=eigenvec_index + 2)
        xs = self.kinetics_model.discretisation_grid.get_rs()
        # shape: (number_cells, num_eigenvectors)
        # average over all directions so that the shape goes to (num_radial, num_eigenvectors)
        num_ray_cells = self.kinetics_model.discretisation_grid.num_radial
        num_angular = self.kinetics_model.discretisation_grid.num_angular
        eigenvecs = eigenvecs.T[eigenvec_index]

        #for i in range(num_angular):
        indices_i_ray = self.kinetics_model.discretisation_grid.get_all_indices_of_slice_j(angular_index)
        eigenvec_i_ray = np.take(eigenvecs, indices_i_ray)
        sns.lineplot(ax=self.ax, x=xs, y=eigenvec_i_ray, **kwargs)
        self.ax.set_ylabel(f"Eigenvector {eigenvec_index}")


class ConvergenceWithAlphaPlot(RepresentationCollection):

    def __init__(self, grid: PolarGrid, alphas: list = None, potential_class = FlatDoubleWellAlpha,
                 potential_parameters = (), kinetics_class = FlatSQRA, kinetics_parameters = ()):
        if alphas is None:
            alphas = np.linspace(0, 25, num=20)
        self.alphas = alphas
        self.all_potentials = []
        for alpha in self.alphas:
            self.all_potentials.append(potential_class(alpha=alpha, *potential_parameters))
        self.grid = grid
        self.all_kinetics = []
        self.kinetics_class = kinetics_class
        self.kinetics_parameters = kinetics_parameters
        super().__init__("")

    def get_all_kinetics(self):
        if not self.all_kinetics:
            for potential in self.all_potentials:
                self.all_kinetics.append(self.kinetics_class(self.grid, potential, *self.kinetics_parameters))
        return self.all_kinetics

    @fig_ax_wrapper
    def plot_population_ray_convergence(self, **kwargs):

        def animate(i):
            pp = PotentialPlot(self.all_potentials[i], self.grid, default_context="talk")
            pp.plot_potential_ray(ax=self.ax, color=colors[i], save=False) #,fig=self.fig,
            pp.plot_population_ray(ax=ax2, color="black", save=False)
            self.ax.set_title(f"alpha={np.round(self.alphas[i], 3)}")
            return self.ax

        ax2 = self.ax.twinx()
        ax2.set_ylim(0, 0.21)
        self.ax.set_ylabel("Potential")
        ax2.set_ylabel("Population")
        colors = pl.cm.jet(np.linspace(0, 1, len(self.alphas)))
        anim = FuncAnimation(self.fig, animate, frames=len(self.alphas), interval=50)
        dpi = kwargs.pop("dpi", 200)
        self._save_animation_type(anim, "population_ray", fps=5, dpi=dpi)
        return anim

    @fig_ax_wrapper
    def plot_its_convergence(self, num_eigenv=3):
        all_kin = self.get_all_kinetics()
        data = np.zeros((len(self.alphas), num_eigenv))
        for i, kin in enumerate(all_kin):
            its = kin.get_its(num_eigenv+1)
            data[i] = its
        columns = []
        for i in range(num_eigenv):
            columns.append(f"ITS {i+1}")
        df = pd.DataFrame(data, columns=columns, index=self.alphas)
        sns.lineplot(data=df, ax=self.ax)
        self.ax.set_yscale("log")

    @fig_ax_wrapper
    def plot_eigenvector_ray_convergence(self, eigenvec_index: int, **kwargs):

        def animate(i):
            pp = PotentialPlot(self.all_potentials[i], self.grid, default_context="talk")
            try:
                self.ax.lines.pop()
                ax2.lines.pop()
            except:
                pass
            pp.plot_potential_ray(ax=self.ax, color="black", save=False) #,fig=self.fig,
            kp = KineticsPlot(self.get_all_kinetics()[i])
            kp.make_one_eigenvector_ray_plot(eigenvec_index=eigenvec_index, ax=ax2, color=colors[i], save=False)
            self.ax.set_title(f"alpha={np.round(self.alphas[i], 3)}")
            return self.ax

        ax2 = self.ax.twinx()
        ax2.set_ylim(-0.21, 0.21)
        colors = pl.cm.jet(np.linspace(0, 1, len(self.alphas)))
        self.ax.set_ylabel("Potential")
        ax2.set_ylabel(f"Eigenvector {eigenvec_index}")
        anim = FuncAnimation(self.fig, animate, frames=len(self.alphas), interval=50)
        dpi = kwargs.pop("dpi", 200)
        self._save_animation_type(anim, f"eigenvector_{eigenvec_index}_ray", fps=5, dpi=dpi)
        return anim

    @fig_ax_wrapper
    def plot_eigenvector_convergence(self, eigenvec_index: int, **kwargs):

        def animate(i):
            try:
                self.ax.clear()
                #self.ax.lines.pop()
            except:
                pass
            kp = KineticsPlot(self.get_all_kinetics()[i])
            kp.make_one_eigenvector_plot(eigenvec_index=eigenvec_index, ax=self.ax, s=1, save=False, norm=norm)
            self.ax.set_title(f"alpha={np.round(self.alphas[i], 3)}")
            return self.ax

        min_value = np.infty
        max_value = -np.infty
        for kin_model in self.get_all_kinetics():
            e_val, e_vec = kin_model.get_eigenval_eigenvec(eigenvec_index+1)
            one_evec = e_vec.T[eigenvec_index]
            for value in one_evec:
                if value > max_value:
                    max_value = value
                if value < min_value:
                    min_value = value
        norm = Normalize(min_value, max_value)
        anim = FuncAnimation(self.fig, animate, frames=len(self.alphas), interval=50)
        dpi = kwargs.pop("dpi", 200)
        self._save_animation_type(anim, f"eigenvector_{eigenvec_index}", fps=5, dpi=dpi)
        return anim

    @fig_ax_wrapper
    def plot_potential_convergence(self, **kwargs):

        def animate(i):
            try:
                self.ax.clear()
                self.ax.lines.pop()
            except:
                pass
            pp = PotentialPlot(self.all_potentials[i], self.grid, default_context="talk")
            pp.plot_potential_circles(ax=self.ax, save=False, norm=norm)
            self.ax.set_title(f"alpha={np.round(self.alphas[i], 3)}")
            return self.ax

        min_value = np.infty
        max_value = -np.infty
        for kin in self.get_all_kinetics():
            for value in kin.get_potentials():
                if value > max_value:
                    max_value = value
                if value < min_value:
                    min_value = value
        norm = Normalize(min_value, max_value)
        anim = FuncAnimation(self.fig, animate, frames=len(self.alphas), interval=50)
        dpi = kwargs.pop("dpi", 200)
        self._save_animation_type(anim, f"circle_potential", fps=5, dpi=dpi)
        return anim


if __name__ == "__main__":
    pg = PolarGrid(r_lim=(0.1, 3.9), num_radial=60, num_angular=5)

    cwap = ConvergenceWithAlphaPlot(pg)
    #cwap.plot_potential_convergence()
    #cwap.plot_eigenvector_convergence(0)
    #cwap.plot_eigenvector_convergence(1)
    #cwap.plot_eigenvector_convergence(2)
    # cwap.plot_its_convergence()
    cwap.plot_population_ray_convergence()
    cwap.plot_eigenvector_ray_convergence(0)
    cwap.plot_eigenvector_ray_convergence(1)
    cwap.plot_eigenvector_ray_convergence(2)

