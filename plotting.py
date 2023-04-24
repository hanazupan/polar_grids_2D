"""
Plot potentials here.
"""
from functools import wraps
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import colors
from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from molgri.plotting.abstract import RepresentationCollection
from molgri.space.utils import normalise_vectors
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import geometric_slerp

from potentials import AnalyticalCircularPotential
from polar_grids import PolarGrid


def plot_voronoi_cells(sv, ax, plot_vertex_points=True):
    #sv.sort_vertices_of_regions()
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


class PotentialPlot(RepresentationCollection):

    """
    This is a collection of plotting methods centered around the AnalyticalCircularPotential object.
    """

    def __init__(self, acp: AnalyticalCircularPotential, grid: PolarGrid, *args, **kwargs):
        data_name = acp.get_name()
        super().__init__(data_name, *args, **kwargs)
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

    def _colorbar_x_y_potential(self, potentials: NDArray):
        # colorbar instead of a legend
        cbar = self._add_colorbar(cmap=self.default_cmap, min_value=np.min(potentials), max_value=np.max(potentials))
        if self.ax.get_legend():
            self.ax.get_legend().remove()

        # labels
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        cbar.ax.set_ylabel("Potential")

    @fig_ax_wrapper
    def plot_one_ray(self, theta: float = 0, **kwargs):
        """
        A plot that shows how potential changes with radial distance for a specific angle theta.
        """
        # calculate potentials for all these different radii and the value of theta
        coords = self.grid.get_radial_grid(theta=theta)
        potential_per_r = self.acp.get_potential(coords)

        color = kwargs.pop("color", "black")
        sns.lineplot(x=coords.T[0], y=potential_per_r, ax=self.ax, color=color, **kwargs)
        self.ax.set_title(r'$\theta=$'+f"{theta}")

    @fig_ax_wrapper
    def plot_colored_circles(self):
        """
        A plot that shows the view from above - concentrical circles of potential values.
        """
        coords = self.grid.get_flattened_polar_coords()
        potentials = self.acp.get_potential(coords)
        xy_coords = self.grid.get_flattened_cartesian_coords()

        sns.scatterplot(x=xy_coords.T[0], y=xy_coords.T[1], ax=self.ax, hue=potentials, palette=self.default_cmap, s=2)

        self._colorbar_x_y_potential(potentials)

    @is_3d_plot
    @fig_ax_wrapper
    def plot_potential_3D(self):
        coord_meshgrid = self.grid.get_polar_meshgrid()
        X, Y = self.grid.get_cartesian_meshgrid()
        potentials = self.acp.get_potential_as_meshgrid(coord_meshgrid[0], coord_meshgrid[1])

        self.ax.plot_surface(X, Y, potentials, cmap=self.default_cmap, linewidth=0, antialiased=False, alpha=0.5)
        self._colorbar_x_y_potential(potentials)


class PolarPlot(RepresentationCollection):

    def __init__(self, pg: PolarGrid, *args, **kwargs):
        self.pg = pg
        data_name = self.pg.get_name()
        super().__init__(data_name, *args, **kwargs)

    @fig_ax_wrapper
    def plot_gridpoints(self):
        points = self.pg.get_flattened_cartesian_coords()
        self.ax.scatter(*points.T, color="black", s=2)

    @fig_ax_wrapper
    def plot_radial_grid(self, **kwargs):
        """
        A plot that shows how potential changes with radial distance for a specific angle theta.
        """
        # calculate potentials for all these different radii and the value of theta
        coords = self.pg.get_radial_grid()

        color = kwargs.pop("color", "black")
        sns.lineplot(x=coords.T[0], y=coords.T[1], ax=self.ax, color=color, markers=True, **kwargs)
        sns.scatterplot(x=coords.T[0], y=coords.T[1], ax=self.ax, color=color, **kwargs)

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


if __name__ == "__main__":
    # from potentials import FlatSymmetricalDoubleWell
    # potential = FlatSymmetricalDoubleWell(12, 4, 7.5)
    # my_grid = PolarGrid(r_lim=(0, 8))
    #
    # pp = PotentialPlot(potential, my_grid, default_context="talk")
    #
    # fig, ax = plt.subplots(1, 2)
    # pp.plot_one_ray(theta=7, fig=fig, ax=ax[0], color="red", save=False)
    # pp.plot_one_ray(theta=22, fig=fig, ax=ax[1], color="blue")
    #
    # pp.plot_colored_circles()
    #
    # pp.plot_potential_3D(animate_rot=True)
    my_pg = PolarGrid(r_lim=(3, 7), num_radial=5, num_angular=15)
    pol_plot = PolarPlot(my_pg, default_complexity_level="empty")
    pol_plot.plot_radial_grid()
    pol_plot.plot_voronoi_cells(numbered=True)

    voronoi_grid = my_pg.get_full_voronoi_grid()
    surf_array = voronoi_grid.get_all_voronoi_surfaces_as_numpy()
    surf_plot = ArrayPlot(surf_array)
    surf_plot.make_heatmap_plot(data_name="surfaces")

    dist_array = voronoi_grid.get_all_distances_between_centers_as_numpy()
    dist_plot = ArrayPlot(surf_array)
    dist_plot.make_heatmap_plot(data_name="dist")

