import matplotlib.pyplot as plt
import numpy as np

def plot_wires(array):
    non_empty_wires = array[array != 0]
    unique_wires = np.unique(non_empty_wires)
    num_wires = int(unique_wires.shape[0] / 3)

    color_map = plt.get_cmap("jet")
    colors = color_map(np.linspace(0, 1, num_wires))

    fig, ax = plt.subplots()
    for i, wire_value in enumerate(unique_wires[::3]):
        wire_indices = np.where((array >= wire_value) & (array <= wire_value + 2))
        x, y = np.transpose(np.column_stack(wire_indices))
        ax.scatter(x, y, c=colors[i], s=50)
    plt.show()

array = np.array([[ 4,  0,  0,  7,  5,  5,  5],
 [ 3,  0,  0,  5,  5,  6,  5],
 [10,  0,  0,  5,  5,  5,  5],
 [ 8,  9, 13, 12, 20, 20,  0],
 [16, 17, 17, 19, 20, 20, 20],
 [14, 17, 17, 17, 22,  0, 21],
 [14, 15,  0, 17, 17, 17, 18]])
plot_wires(array)