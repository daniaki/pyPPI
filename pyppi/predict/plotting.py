"""
This module contains the two functions used to make the heat-map plots
and the threshold curves.
"""


import numpy as np
import matplotlib
matplotlib.use("tkagg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import pandas as pd


__all__ = [
    "plot_heatmaps",
    "plot_threshold_curve"
]


def plot_heatmaps(path, labels, correlation_matrix, similarity_matrix,
                  dpi=350, format='jpg'):
    plt.style.use('default')
    ticklabels = [l.capitalize() for l in labels]

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(1, 2)
    gs.update(**dict(wspace=0.2))
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1], sharey=ax1)

    #-- Label sim
    ax1.set_xticks(range(0, 18))
    ax1.set_xticklabels(
        ticklabels, rotation=45, ha='right', fontdict={'size': 14})

    ax1.set_yticks(range(0, 18))
    ax1.set_yticklabels(ticklabels, fontdict={'size': 14})
    ax1.text(0, 1.01, '(a)', transform=ax1.transAxes, size=12)

    im1 = ax1.imshow(
        correlation_matrix, interpolation='none', cmap='RdBu', vmin=-1, vmax=1)
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax=cax1, ticks=[-1., -0.5, 0., 0.5, 1.])

    #-- Jacard
    ax2.set_xticks(range(0, 18))
    ax2.set_xticklabels(
        ticklabels, rotation=45, ha='right', fontdict={'size': 14})

    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.text(0, 1.01, '(b)', transform=ax2.transAxes, size=12)

    im2 = ax2.imshow(
        similarity_matrix, interpolation='none', cmap='RdPu', vmin=0, vmax=1)
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im2, cax=cax2, ticks=[0., .5, 1.])

    plt.savefig(
        path, format=format, dpi=dpi, bbox_inches='tight', pad_inches=0)

    return fig, ax1, ax2


def plot_threshold_curve(path, thresholds, proportions, dpi=350, format='jpg'):
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1)
    ax.plot(thresholds, proportions, 's-', color='black', linewidth='1', )
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels([0, 0.5, 1.0], fontdict={'size': 20})
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels([0, 0.5, 1.0], fontdict={'size': 20})
    ax.set_ylabel("Proportion classified", fontdict={'size': 20})
    ax.set_xlabel("Threshold probability", fontdict={'size': 20})

    plt.tight_layout()
    plt.savefig(
        path, format=format, dpi=dpi,
        bbox_inches='tight', pad_inches=0
    )
    return fig, ax
