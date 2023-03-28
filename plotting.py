"""
Plot potentials here.
"""
from functools import wraps

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from potentials import AnalyticalCircularPotential
from polar_grids import PolarGrid


def fig_ax_wrapper(my_method):

    @wraps(my_method)
    def decorated(*args, **kwargs):
        if "ax" not in kwargs:
            kwargs["ax"] = None
        if "fig" not in kwargs:
            kwargs["fig"] = None
        func_value = my_method(*args, **kwargs)
        return func_value
    return decorated


class PotentialPlot:

    def __init__(self, acp: AnalyticalCircularPotential, grid: PolarGrid):
        self.acp = acp
        self.grid = grid

    @fig_ax_wrapper
    def plot_one_ray(self, theta: float = 0, **kwargs):
        """
        A plot that shows how potential changes with radial distance for a specific angle theta.
        """

        # calculate potentials for all these different radii and the value of theta
        coords = self.grid.get_radial_grid(theta=theta)
        potential_per_r = self.acp.get_potential(coords)

        color = kwargs.pop("color", "black")
        sns.lineplot(x=coords.T[0], y=potential_per_r, ax=kwargs["ax"], color=color)

    @fig_ax_wrapper
    def plot_colored_circles(self, **kwargs):
        """
        A plot that shows the view from above - concentrical circles of potential values.
        """
        coords = self.grid.get_flattened_polar_coords()
        potentials = self.acp.get_potential(coords)
        xy_coords = self.grid.get_flattened_cartesian_coords()

        sc = sns.scatterplot(x=xy_coords.T[0], y=xy_coords.T[1], ax=kwargs["ax"], hue=potentials, palette="coolwarm")
        plt.gca().get_legend().remove()
        #plt.gca().figure.colorbar(sc)

    @fig_ax_wrapper
    def plot_potential_3D(self, **kwargs):
        coord_meshgrid = self.grid.get_polar_meshgrid()
        X, Y = self.grid.get_cartesian_meshgrid()
        potentials = self.acp.get_potential_as_meshgrid(coord_meshgrid[0], coord_meshgrid[1])

        if kwargs["ax"] is None or kwargs["fig"] is None:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        else:
            fig = kwargs["fig"]
            ax = kwargs["ax"]

        ax.plot_surface(X, Y, potentials, cmap="coolwarm",
                        linewidth=0, antialiased=False)




if __name__ == "__main__":
    from potentials import FlatSymmetricalDoubleWell
    potential = FlatSymmetricalDoubleWell(12, 4, 7.5)
    my_grid = PolarGrid(r_lim=(0, 8))

    pp = PotentialPlot(potential, my_grid)

    fig, ax = plt.subplots(1, 2)
    pp.plot_one_ray(theta=7, r_lim=(0, 8), fig=fig, ax=ax[0], color="red")
    pp.plot_one_ray(theta=22, r_lim=(0, 8), fig=fig, ax=ax[1], color="blue")
    plt.show()
    fig, ax = plt.subplots(1, 1)
    pp.plot_colored_circles()
    plt.show()
    pp.plot_potential_3D()
    plt.show()
