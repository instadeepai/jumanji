import math
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid


def plot_heatmap(scores: np.ndarray):
    # TODO (Marta): Add saving capability
    fig, ax = plt.subplots()
    im = ax.imshow(scores, cmap='Purples', interpolation='nearest')
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.show()


def plot_comparison_heatmap(
    list_of_scores: List[np.ndarray],
    list_of_titles: List[str],
    num_agents: int,
    number_of_boards_averaged: int,
):
    n_rows = max(math.ceil(len(list_of_scores) / 3), 1)
    n_columns = 3
    fig = plt.figure(figsize=(6, 2 * n_rows + 0.5))
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(n_rows, n_columns),
                    axes_pad=0.3,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.1
                    )
    plt.suptitle(f"Scores per Cell Averaged on {number_of_boards_averaged} Boards with {num_agents} wires ", fontsize=12, y=0.98)

    all_scores = np.array(list_of_scores).flatten()
    min_score = all_scores.min()
    max_score = all_scores.max()

    for i, ax in enumerate(grid):

        if i >= len(list_of_scores):
            fig.delaxes(ax)
        else:
            print(list_of_scores[i])
            # normalised_board = normalize(list_of_scores[i], min_score, max_score)
            ax.set_title(list_of_titles[i])
            im = ax.imshow(list_of_scores[i], cmap='Purples')
            im.set_clim(min_score, max_score)
            ax.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)

    cbar = ax.cax.colorbar(im)
    cbar = grid.cbar_axes[0].colorbar(im)
    plt.tight_layout()
    plt.show()
